// Multiplexed neural networks: a second network computes per-sample weights for the first one.
//
// Main network (sizes n_0 .. n_L), input a_0 = x, ã_i = [a_i; 1]:
//     z_{i+1} = W_eff,i(x)^T ã_i,     W_eff,i(x) = W_i + sum_l g_{i,l}(x) M_{i,l}
//     a_{i+1} = tanh(z_{i+1})          (hidden layers; the last layer gives logits or linear outputs)
// Control network (sizes m_0 = n_0 .. m_K), b_0 = x:
//     b_{i+1} = tanh(V_i^T [b_i; 1]),  and g_i = b_i modulates main layer i -> i+1 (for i <= K).
// So the first layer is bilinear in x (z_1 = W_0^T x̃ + sum_l x_l M_{0,l}^T x̃) and deeper layers are
// modulated by learned features of x. Without a control network this is a plain MLP.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mnn {

enum class Loss { SoftmaxCrossEntropy, MeanSquaredError };

struct Config {
    std::vector<int> main;     // n_0 .. n_L (n_0 = number of inputs, n_L = classes or outputs)
    std::vector<int> control;  // m_0 .. m_K with m_0 = n_0, K < L; empty = plain MLP
    Loss loss = Loss::SoftmaxCrossEntropy;
    double modInitScale = 1.0;  // M_{i,l} ~ N(0, scale^2 / ((n_i + 1) * m_i)) ... see init()
    uint64_t seed = 1;
};

// Dense parameter block with its gradient.
struct Param {
    std::vector<double> w, g;
    void resize(size_t n) { w.assign(n, 0.0), g.assign(n, 0.0); }
    size_t size() const { return w.size(); }
};

class Model {
public:
    explicit Model(Config cfg) : c_(std::move(cfg)) {
        const auto &n = c_.main;
        if (n.size() < 2) throw std::invalid_argument("the main network needs at least an input and an output layer");
        for (int s : n)
            if (s < 1) throw std::invalid_argument("layer sizes must be positive");
        if (!c_.control.empty()) {
            if (c_.control[0] != n[0]) throw std::invalid_argument("control network input size must equal main input size");
            if (c_.control.size() > n.size() - 1)
                throw std::invalid_argument("control network has more layers than the main network has weight matrices");
            for (int s : c_.control)
                if (s < 1) throw std::invalid_argument("layer sizes must be positive");
        }
        L_ = (int)n.size() - 1;
        K_ = (int)c_.control.size();  // number of modulated main layers (control layers 0..K-1)
        W_.resize(L_);
        M_.resize(K_);
        V_.resize(K_ > 0 ? K_ - 1 : 0);
        for (int i = 0; i < L_; i++) W_[i].resize((size_t)(n[i] + 1) * n[i + 1]);
        for (int i = 0; i < K_; i++) M_[i].resize((size_t)c_.control[i] * (n[i] + 1) * n[i + 1]);
        for (int i = 0; i + 1 < K_; i++) V_[i].resize((size_t)(c_.control[i] + 1) * c_.control[i + 1]);
        init(c_.seed);
    }

    const Config &config() const { return c_; }
    int inputs() const { return c_.main.front(); }
    int outputs() const { return c_.main.back(); }
    bool multiplexed() const { return K_ > 0; }

    std::vector<Param *> params() {
        std::vector<Param *> p;
        for (auto &x : W_) p.push_back(&x);
        for (auto &x : M_) p.push_back(&x);
        for (auto &x : V_) p.push_back(&x);
        return p;
    }
    size_t parameterCount() {
        size_t s = 0;
        for (auto *p : params()) s += p->size();
        return s;
    }

    void init(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        const auto &n = c_.main;
        // Xavier/Glorot for W and V, biases zero
        for (int i = 0; i < L_; i++) {
            const double s = std::sqrt(2.0 / (n[i] + n[i + 1]));
            for (int k = 0; k < n[i]; k++)
                for (int j = 0; j < n[i + 1]; j++) W_[i].w[idxW(i, k, j)] = s * nd(rng);
        }
        for (int i = 0; i + 1 < K_; i++) {
            const auto &m = c_.control;
            const double s = std::sqrt(2.0 / (m[i] + m[i + 1]));
            for (int k = 0; k < m[i]; k++)
                for (int j = 0; j < m[i + 1]; j++) V_[i].w[(size_t)k * m[i + 1] + j] = s * nd(rng);
        }
        // M: the modulation sum_l g_l M_l has about the variance of one Xavier matrix times modInitScale^2
        for (int i = 0; i < K_; i++) {
            const double s = c_.modInitScale * std::sqrt(2.0 / (n[i] + n[i + 1])) / std::sqrt((double)c_.control[i]);
            for (auto &v : M_[i].w) v = s * nd(rng);
        }
        zeroGrad();
    }

    void zeroGrad() {
        for (auto *p : params()) std::fill(p->g.begin(), p->g.end(), 0.0);
    }

    // Forward pass for one sample; returns the outputs (logits for cross-entropy). Keeps the activations
    // needed by backward().
    const std::vector<double> &forward(const std::vector<double> &x) {
        const auto &n = c_.main;
        if ((int)x.size() != n[0]) throw std::invalid_argument("input has the wrong size");
        a_.resize(L_ + 1);
        b_.resize(K_);
        a_[0] = x;
        if (K_ > 0) b_[0] = x;
        for (int i = 1; i < K_; i++) {  // control network
            const int mi = c_.control[i - 1], mo = c_.control[i];
            b_[i].assign(mo, 0.0);
            for (int k = 0; k <= mi; k++) {
                const double v = k < mi ? b_[i - 1][k] : 1.0;
                const double *row = &V_[i - 1].w[(size_t)k * mo];
                for (int j = 0; j < mo; j++) b_[i][j] += v * row[j];
            }
            for (auto &v : b_[i]) v = std::tanh(v);
        }
        for (int i = 0; i < L_; i++) {  // main network
            const int ni = n[i], no = n[i + 1];
            std::vector<double> z(no, 0.0);
            for (int k = 0; k <= ni; k++) {
                const double v = k < ni ? a_[i][k] : 1.0;
                if (v == 0.0) continue;
                const double *row = &W_[i].w[idxW(i, k, 0)];
                for (int j = 0; j < no; j++) z[j] += v * row[j];
                if (i < K_) {
                    for (int l = 0; l < c_.control[i]; l++) {
                        const double gv = b_[i][l] * v;
                        const double *mrow = &M_[i].w[idxM(i, l, k, 0)];
                        for (int j = 0; j < no; j++) z[j] += gv * mrow[j];
                    }
                }
            }
            if (i + 1 < L_)
                for (auto &v : z) v = std::tanh(v);
            a_[i + 1] = std::move(z);
        }
        return a_[L_];
    }

    // Loss of the last forward pass. Cross-entropy: target[0] is the class index. MSE: target = values.
    double loss(const std::vector<double> &target) const {
        const auto &o = a_[L_];
        if (c_.loss == Loss::SoftmaxCrossEntropy) {
            const double mx = *std::max_element(o.begin(), o.end());
            double s = 0;
            for (double v : o) s += std::exp(v - mx);
            return -(o[classOf(target)] - mx - std::log(s));
        }
        double s = 0;
        for (size_t j = 0; j < o.size(); j++) s += 0.5 * (o[j] - target[j]) * (o[j] - target[j]);
        return s;
    }

    // Accumulate d(scale * loss)/d(params) for the last forward pass into the gradients.
    void backward(const std::vector<double> &target, double scale = 1.0) {
        const auto &n = c_.main;
        const auto &o = a_[L_];
        std::vector<double> delta(o.size());
        if (c_.loss == Loss::SoftmaxCrossEntropy) {
            const double mx = *std::max_element(o.begin(), o.end());
            double s = 0;
            for (size_t j = 0; j < o.size(); j++) s += (delta[j] = std::exp(o[j] - mx));
            for (auto &d : delta) d /= s;
            delta[classOf(target)] -= 1.0;
        } else {
            for (size_t j = 0; j < o.size(); j++) delta[j] = o[j] - target[j];
        }
        for (auto &d : delta) d *= scale;

        std::vector<std::vector<double>> gb(K_);  // dL/db_i from the modulation
        for (int i = L_ - 1; i >= 0; i--) {
            const int ni = n[i], no = n[i + 1];
            std::vector<double> ga(ni, 0.0);  // dL/da_i
            if (i < K_) gb[i].assign(c_.control[i], 0.0);
            for (int k = 0; k <= ni; k++) {
                const double v = k < ni ? a_[i][k] : 1.0;
                double *gW = &W_[i].g[idxW(i, k, 0)];
                const double *w = &W_[i].w[idxW(i, k, 0)];
                double back = 0;
                for (int j = 0; j < no; j++) gW[j] += v * delta[j], back += w[j] * delta[j];
                if (i < K_) {
                    for (int l = 0; l < c_.control[i]; l++) {
                        const double gl = b_[i][l];
                        double *gM = &M_[i].g[idxM(i, l, k, 0)];
                        const double *m = &M_[i].w[idxM(i, l, k, 0)];
                        double md = 0;  // (M_l delta)_k
                        for (int j = 0; j < no; j++) gM[j] += gl * v * delta[j], md += m[j] * delta[j];
                        back += gl * md;
                        gb[i][l] += v * md;
                    }
                }
                if (k < ni) ga[k] = back;
            }
            if (i > 0) {
                for (int k = 0; k < ni; k++) ga[k] *= 1.0 - a_[i][k] * a_[i][k];
                delta = std::move(ga);
            }
        }
        // control network: b_i = tanh(V_{i-1}^T [b_{i-1}; 1]) for i >= 1; b_0 = x has no parameters
        for (int i = K_ - 1; i >= 1; i--) {
            const int mi = c_.control[i - 1], mo = c_.control[i];
            std::vector<double> d(mo);
            for (int j = 0; j < mo; j++) d[j] = gb[i][j] * (1.0 - b_[i][j] * b_[i][j]);
            for (int k = 0; k <= mi; k++) {
                const double v = k < mi ? b_[i - 1][k] : 1.0;
                double *gV = &V_[i - 1].g[(size_t)k * mo];
                const double *w = &V_[i - 1].w[(size_t)k * mo];
                double back = 0;
                for (int j = 0; j < mo; j++) gV[j] += v * d[j], back += w[j] * d[j];
                if (k < mi) gb[i - 1][k] += back;
            }
        }
    }

    // Per-sample effective weight matrix of main layer i (for inspection): (n_i + 1) x n_{i+1}, row-major.
    std::vector<double> effectiveWeights(const std::vector<double> &x, int i) {
        forward(x);
        std::vector<double> e = W_.at(i).w;
        if (i < K_)
            for (int l = 0; l < c_.control[i]; l++)
                for (size_t q = 0; q < e.size(); q++) e[q] += b_[i][l] * M_[i].w[(size_t)l * e.size() + q];
        return e;
    }

    int predict(const std::vector<double> &x) {
        const auto &o = forward(x);
        return (int)(std::max_element(o.begin(), o.end()) - o.begin());
    }

private:
    size_t idxW(int i, int k, int j) const { return (size_t)k * c_.main[i + 1] + j; }
    size_t idxM(int i, int l, int k, int j) const {
        return ((size_t)l * (c_.main[i] + 1) + k) * c_.main[i + 1] + j;
    }
    int classOf(const std::vector<double> &t) const {
        const int c = (int)t.at(0);
        if (c < 0 || c >= outputs()) throw std::out_of_range("class index out of range");
        return c;
    }

    Config c_;
    int L_ = 0, K_ = 0;
    std::vector<Param> W_, M_, V_;
    std::vector<std::vector<double>> a_, b_;
};

// Adam, or SGD with (heavy-ball) momentum when adam = false.
struct Optimizer {
    double lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, momentum = 0.9, weightDecay = 0.0;
    bool adam = true;
    long t = 0;
    std::vector<std::vector<double>> m, v;

    void step(Model &model) {
        auto ps = model.params();
        if (m.size() != ps.size()) {
            m.clear(), v.clear();
            for (auto *p : ps) m.emplace_back(p->size(), 0.0), v.emplace_back(p->size(), 0.0);
        }
        t++;
        const double c1 = 1.0 - std::pow(beta1, (double)t), c2 = 1.0 - std::pow(beta2, (double)t);
        for (size_t q = 0; q < ps.size(); q++) {
            auto &p = *ps[q];
            for (size_t i = 0; i < p.size(); i++) {
                const double g = p.g[i] + weightDecay * p.w[i];
                if (adam) {
                    m[q][i] = beta1 * m[q][i] + (1 - beta1) * g;
                    v[q][i] = beta2 * v[q][i] + (1 - beta2) * g * g;
                    p.w[i] -= lr * (m[q][i] / c1) / (std::sqrt(v[q][i] / c2) + eps);
                } else {
                    m[q][i] = momentum * m[q][i] + g;  // velocity = previous step direction
                    p.w[i] -= lr * m[q][i];
                }
            }
        }
        model.zeroGrad();
    }
};

}  // namespace mnn

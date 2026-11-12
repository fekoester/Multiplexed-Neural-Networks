#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <random>

#include "mnn/data.hpp"
#include "mnn/model.hpp"

using namespace mnn;

// Compare backward() with central finite differences for every parameter.
static double maxRelativeGradError(Config cfg, int samples = 3) {
    Model m(cfg);
    std::mt19937_64 rng(cfg.seed + 100);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<double>> xs, ts;
    for (int s = 0; s < samples; s++) {
        std::vector<double> x((size_t)m.inputs());
        for (auto &v : x) v = nd(rng);
        xs.push_back(x);
        if (cfg.loss == Loss::SoftmaxCrossEntropy) ts.push_back({double(s % m.outputs())});
        else {
            std::vector<double> t((size_t)m.outputs());
            for (auto &v : t) v = nd(rng);
            ts.push_back(t);
        }
    }
    auto total = [&]() {
        double L = 0;
        for (int s = 0; s < samples; s++) m.forward(xs[s]), L += m.loss(ts[s]);
        return L;
    };
    m.zeroGrad();
    for (int s = 0; s < samples; s++) m.forward(xs[s]), m.backward(ts[s]);
    double worst = 0;
    const double h = 1e-6;
    for (auto *p : m.params()) {
        for (size_t i = 0; i < p->size(); i++) {
            const double keep = p->w[i];
            p->w[i] = keep + h;
            const double up = total();
            p->w[i] = keep - h;
            const double down = total();
            p->w[i] = keep;
            const double num = (up - down) / (2 * h);
            const double err = std::abs(num - p->g[i]) / std::max(1e-6, std::abs(num) + std::abs(p->g[i]));
            worst = std::max(worst, err);
        }
    }
    return worst;
}

TEST_CASE("gradients match finite differences") {
    struct Case {
        std::vector<int> main, control;
        Loss loss;
    };
    const std::vector<Case> cases = {
        {{4, 5, 3}, {}, Loss::SoftmaxCrossEntropy},                  // plain MLP
        {{4, 5, 3}, {4}, Loss::SoftmaxCrossEntropy},                 // bilinear first layer only
        {{4, 5, 3}, {4, 3}, Loss::SoftmaxCrossEntropy},              // both layers modulated
        {{3, 6, 5, 2}, {3, 4, 2}, Loss::MeanSquaredError},           // the repo's original layout, smaller
        {{3, 6, 5, 2}, {3, 4}, Loss::MeanSquaredError},              // control shallower than main
        {{5, 4, 4, 4, 3}, {5, 2, 3}, Loss::SoftmaxCrossEntropy},     // deeper main network
    };
    for (const auto &c : cases) {
        Config cfg;
        cfg.main = c.main, cfg.control = c.control, cfg.loss = c.loss, cfg.seed = 7;
        CAPTURE(c.main.size());
        CAPTURE(c.control.size());
        CHECK(maxRelativeGradError(cfg) < 1e-5);
    }
}

TEST_CASE("without a control network the model is an MLP; with one, weights depend on the input") {
    Config cfg;
    cfg.main = {3, 4, 2};
    Model mlp(cfg);
    CHECK_FALSE(mlp.multiplexed());
    CHECK(mlp.parameterCount() == 4u * 4 + 5 * 2);
    auto e1 = mlp.effectiveWeights({1, 2, 3}, 0), e2 = mlp.effectiveWeights({-1, 0, 5}, 0);
    CHECK(e1 == e2);

    cfg.control = {3, 5};
    Model mpx(cfg);
    CHECK(mpx.multiplexed());
    CHECK(mpx.parameterCount() == 4u * 4 + 5 * 2 + 3 * 4 * 4 + 5 * 5 * 2 + 4 * 5);
    e1 = mpx.effectiveWeights({1, 2, 3}, 0), e2 = mpx.effectiveWeights({-1, 0, 5}, 0);
    CHECK(e1 != e2);
}

TEST_CASE("the first modulated layer is bilinear in the input") {
    // With W_0 = 0, z_1 = sum_l x_l M_l^T [x; 1]: scaling x by s scales the quadratic part by s^2
    // and the bias part by s, so z(2x) = 4 q(x) + 2 r(x) and z(x) = q(x) + r(x), z(-x) = q(x) - r(x).
    Config cfg;
    cfg.main = {3, 4};
    cfg.control = {3};
    cfg.loss = Loss::MeanSquaredError;
    Model m(cfg);
    for (auto *p : m.params())
        if (p->size() == 4u * 4) std::fill(p->w.begin(), p->w.end(), 0.0);  // W_0
    const std::vector<double> x = {0.3, -1.2, 0.7}, x2 = {0.6, -2.4, 1.4}, xn = {-0.3, 1.2, -0.7};
    const auto z = m.forward(x), z2 = m.forward(x2), zn = m.forward(xn);
    for (size_t j = 0; j < z.size(); j++) {
        const double q = 0.5 * (z[j] + zn[j]), r = 0.5 * (z[j] - zn[j]);
        CHECK(z2[j] == doctest::Approx(4 * q + 2 * r));
    }
}

TEST_CASE("invalid configurations are rejected") {
    Config cfg;
    cfg.main = {3};
    CHECK_THROWS_AS(Model{cfg}, std::invalid_argument);
    cfg.main = {3, 4, 2};
    cfg.control = {2};
    CHECK_THROWS_AS(Model{cfg}, std::invalid_argument);
    cfg.control = {3, 4, 4};
    CHECK_THROWS_AS(Model{cfg}, std::invalid_argument);
    cfg.control = {3};
    Model m(cfg);
    CHECK_THROWS_AS(m.forward({1, 2}), std::invalid_argument);
}

TEST_CASE("both models learn XOR, deterministically") {
    for (bool multiplexed : {false, true}) {
        Config cfg;
        cfg.main = {2, 4, 2};
        if (multiplexed) cfg.main = {2, 2}, cfg.control = {2};  // no hidden layer: XOR needs the product term
        cfg.seed = 3;
        auto run = [&]() {
            Model m(cfg);
            Optimizer opt;
            opt.lr = 0.05;
            Split s = makeXor();
            EpochStats last;
            train(m, opt, s, 300, 4, 1, [&](const EpochStats &e) { last = e; });
            return last;
        };
        const EpochStats a = run(), b = run();
        CAPTURE(multiplexed);
        CHECK(a.testAccuracy == 1.0);
        CHECK(a.testLoss < 0.05);
        CHECK(a.testLoss == b.testLoss);
    }
}

TEST_CASE("a linear model cannot learn XOR") {
    Config cfg;
    cfg.main = {2, 2};
    Model m(cfg);
    Optimizer opt;
    opt.lr = 0.05;
    Split s = makeXor();
    EpochStats last;
    train(m, opt, s, 300, 4, 1, [&](const EpochStats &e) { last = e; });
    CHECK(last.testAccuracy < 1.0);
}

TEST_CASE("SGD with momentum steps against the gradient") {
    Config cfg;
    cfg.main = {2, 1};
    cfg.loss = Loss::MeanSquaredError;
    Model m(cfg);
    Optimizer opt;
    opt.adam = false;
    opt.lr = 0.1;
    double prev = 1e9;
    for (int it = 0; it < 200; it++) {
        m.forward({1.0, -1.0});
        prev = m.loss({0.5});
        m.backward({0.5});
        opt.step(m);
    }
    CHECK(prev < 1e-8);
}

TEST_CASE("gated data: labels depend on the context") {
    Split s = makeGated(4, 6, 3, 400, 100, 5);
    CHECK(s.inputs == 10);
    CHECK(s.train.size() == 400);
    for (const auto &y : s.train.y) CHECK((y[0] >= 0 && y[0] < 3));
    for (const auto &x : s.train.x) CHECK(x[0] + x[1] + x[2] + x[3] == 1.0);
}

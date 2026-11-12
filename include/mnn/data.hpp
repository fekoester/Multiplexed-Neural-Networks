// Datasets and the training loop.
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.hpp"

namespace mnn {

struct Dataset {
    std::vector<std::vector<double>> x, y;  // y = {class} for classification
    size_t size() const { return x.size(); }
};

struct Split {
    Dataset train, test;
    int inputs = 0, outputs = 0;
    Loss loss = Loss::SoftmaxCrossEntropy;
};

// Standardise every input feature with the training mean and standard deviation.
inline void standardise(Split &s) {
    const size_t d = (size_t)s.inputs;
    std::vector<double> mu(d, 0.0), sd(d, 0.0);
    for (const auto &x : s.train.x)
        for (size_t j = 0; j < d; j++) mu[j] += x[j];
    for (auto &v : mu) v /= (double)s.train.size();
    for (const auto &x : s.train.x)
        for (size_t j = 0; j < d; j++) sd[j] += (x[j] - mu[j]) * (x[j] - mu[j]);
    for (auto &v : sd) v = std::sqrt(v / (double)s.train.size()) + 1e-12;
    for (auto *set : {&s.train, &s.test})
        for (auto &x : set->x)
            for (size_t j = 0; j < d; j++) x[j] = (x[j] - mu[j]) / sd[j];
}

// UCI letter recognition (letter-recognition.data): "T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8".
// The letter is the first column. Standard split: first 16000 rows train, last 4000 test.
inline Split loadLetters(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path + " (run scripts/get_letter_data.sh)");
    Dataset all;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] < 'A' || line[0] > 'Z') continue;
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        std::vector<double> x;
        while (std::getline(ss, cell, ',')) x.push_back(std::stod(cell));
        if (x.size() != 16) throw std::runtime_error("unexpected row in " + path + ": " + line);
        all.x.push_back(std::move(x));
        all.y.push_back({double(line[0] - 'A')});
    }
    if (all.size() < 1000) throw std::runtime_error(path + " has too few rows");
    Split s;
    s.inputs = 16, s.outputs = 26;
    const size_t ntrain = all.size() * 4 / 5;
    s.train.x.assign(all.x.begin(), all.x.begin() + ntrain);
    s.train.y.assign(all.y.begin(), all.y.begin() + ntrain);
    s.test.x.assign(all.x.begin() + ntrain, all.x.end());
    s.test.y.assign(all.y.begin() + ntrain, all.y.end());
    standardise(s);
    return s;
}

// Context-gated classification: x = [one-hot context c (C values); v ~ N(0, I_d)], label = argmax_k (A_c v)_k
// with a different random matrix A_c per context. The decision boundary is linear in v but its weights depend
// on c -- the product of context and input is exactly what a multiplicative interaction provides.
inline Split makeGated(int contexts, int dim, int classes, size_t ntrain, size_t ntest, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<std::vector<double>> A((size_t)contexts, std::vector<double>((size_t)classes * dim));
    for (auto &a : A)
        for (auto &v : a) v = nd(rng);
    std::uniform_int_distribution<int> pc(0, contexts - 1);
    auto sample = [&](Dataset &d, size_t n) {
        for (size_t s = 0; s < n; s++) {
            const int c = pc(rng);
            std::vector<double> x((size_t)contexts + dim, 0.0);
            x[(size_t)c] = 1.0;
            for (int j = 0; j < dim; j++) x[(size_t)contexts + j] = nd(rng);
            int best = 0;
            double bv = -1e300;
            for (int k = 0; k < classes; k++) {
                double z = 0;
                for (int j = 0; j < dim; j++) z += A[(size_t)c][(size_t)k * dim + j] * x[(size_t)contexts + j];
                if (z > bv) bv = z, best = k;
            }
            d.x.push_back(std::move(x));
            d.y.push_back({double(best)});
        }
    };
    Split s;
    s.inputs = contexts + dim, s.outputs = classes;
    sample(s.train, ntrain);
    sample(s.test, ntest);
    return s;
}

// XOR on {-1, 1}^2 (all four points in train and test).
inline Split makeXor() {
    Split s;
    s.inputs = 2, s.outputs = 2;
    for (int a : {-1, 1})
        for (int b : {-1, 1}) {
            s.train.x.push_back({double(a), double(b)});
            s.train.y.push_back({double(a != b)});
        }
    s.test = s.train;
    return s;
}

struct EpochStats {
    int epoch = 0;
    double trainLoss = 0, testLoss = 0, testAccuracy = 0;
};

inline EpochStats evaluate(Model &m, const Dataset &d) {
    EpochStats e;
    size_t correct = 0;
    for (size_t i = 0; i < d.size(); i++) {
        m.forward(d.x[i]);
        e.testLoss += m.loss(d.y[i]);
        if (m.config().loss == Loss::SoftmaxCrossEntropy && m.predict(d.x[i]) == (int)d.y[i][0]) correct++;
    }
    e.testLoss /= (double)d.size();
    e.testAccuracy = double(correct) / (double)d.size();
    return e;
}

// Mini-batch training with a fixed shuffling seed. Calls onEpoch(stats) after every epoch.
template <class Callback>
void train(Model &m, Optimizer &opt, const Split &s, int epochs, int batch, uint64_t seed, Callback onEpoch) {
    std::mt19937_64 rng(seed);
    std::vector<size_t> order(s.train.size());
    std::iota(order.begin(), order.end(), 0);
    m.zeroGrad();
    for (int ep = 1; ep <= epochs; ep++) {
        std::shuffle(order.begin(), order.end(), rng);
        double total = 0;
        for (size_t start = 0; start < order.size(); start += (size_t)batch) {
            const size_t end = std::min(order.size(), start + (size_t)batch);
            const double scale = 1.0 / double(end - start);
            for (size_t q = start; q < end; q++) {
                m.forward(s.train.x[order[q]]);
                total += m.loss(s.train.y[order[q]]);
                m.backward(s.train.y[order[q]], scale);
            }
            opt.step(m);
        }
        EpochStats e = evaluate(m, s.test);
        e.epoch = ep;
        e.trainLoss = total / (double)order.size();
        onEpoch(e);
    }
}

}  // namespace mnn

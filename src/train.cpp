// Train one model and print the learning curve.
//
//   mnn_train --task letter --main 16,32,26 --control 16,8 --epochs 30
//   mnn_train --task gated  --main 12,64,4               (no --control: plain MLP)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mnn/data.hpp"
#include "mnn/model.hpp"

using namespace mnn;

static std::vector<int> parseSizes(const std::string &s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string cell;
    while (std::getline(ss, cell, ',')) v.push_back(std::stoi(cell));
    return v;
}

static void usage() {
    std::cout << "usage: mnn_train [--task letter|gated|xor] [--data letter-recognition.data]\n"
                 "                 [--main 16,32,26] [--control 16,8] [--epochs 30] [--batch 32]\n"
                 "                 [--lr 1e-3] [--sgd] [--momentum 0.9] [--weight-decay 0] [--mod-init 1]\n"
                 "                 [--seed 1] [--csv curve.csv]\n"
                 "--main: sizes of the main network (first = inputs, last = classes); defaults per task\n"
                 "--control: sizes of the control network (first = inputs); omit for a plain MLP\n";
}

int main(int argc, char **argv) {
    std::string task = "letter", data = "letter-recognition.data", csv, mainSizes, controlSizes;
    int epochs = 30, batch = 32;
    double lr = 1e-3, momentum = 0.9, wd = 0.0, modInit = 1.0;
    bool sgd = false;
    uint64_t seed = 1;
    try {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "-h" || a == "--help") return usage(), 0;
            else if (a == "--task") task = next();
            else if (a == "--data") data = next();
            else if (a == "--main") mainSizes = next();
            else if (a == "--control") controlSizes = next();
            else if (a == "--epochs") epochs = std::stoi(next());
            else if (a == "--batch") batch = std::stoi(next());
            else if (a == "--lr") lr = std::stod(next());
            else if (a == "--momentum") momentum = std::stod(next());
            else if (a == "--weight-decay") wd = std::stod(next());
            else if (a == "--mod-init") modInit = std::stod(next());
            else if (a == "--sgd") sgd = true;
            else if (a == "--seed") seed = std::stoull(next());
            else if (a == "--csv") csv = next();
            else throw std::invalid_argument("unknown argument " + a);
        }
        if (epochs < 1 || batch < 1) throw std::invalid_argument("--epochs and --batch must be positive");

        Split s;
        std::vector<int> defMain;
        if (task == "letter") s = loadLetters(data), defMain = {16, 32, 26};
        else if (task == "gated") s = makeGated(8, 8, 4, 8000, 4000, 12345), defMain = {16, 64, 4};
        else if (task == "xor") s = makeXor(), defMain = {2, 4, 2};
        else throw std::invalid_argument("unknown task " + task);

        Config cfg;
        cfg.main = mainSizes.empty() ? defMain : parseSizes(mainSizes);
        if (!controlSizes.empty()) cfg.control = parseSizes(controlSizes);
        cfg.loss = s.loss;
        cfg.seed = seed;
        cfg.modInitScale = modInit;
        if (cfg.main.front() != s.inputs || cfg.main.back() != s.outputs)
            throw std::invalid_argument("--main must start with " + std::to_string(s.inputs) + " inputs and end with " +
                                        std::to_string(s.outputs) + " outputs for task " + task);
        Model m(cfg);
        Optimizer opt;
        opt.lr = lr, opt.adam = !sgd, opt.momentum = momentum, opt.weightDecay = wd;

        std::printf("task %s, %zu train / %zu test, %s, %zu parameters\n", task.c_str(), s.train.size(),
                    s.test.size(), m.multiplexed() ? "multiplexed" : "MLP", m.parameterCount());
        std::ofstream out;
        if (!csv.empty()) {
            out.open(csv);
            out << "epoch,train_loss,test_loss,test_accuracy\n";
        }
        const auto t0 = std::chrono::steady_clock::now();
        train(m, opt, s, epochs, batch, seed + 1000, [&](const EpochStats &e) {
            std::printf("epoch %3d  train loss %.4f  test loss %.4f  test acc %.4f\n", e.epoch, e.trainLoss,
                        e.testLoss, e.testAccuracy);
            std::fflush(stdout);
            if (out) out << e.epoch << ',' << e.trainLoss << ',' << e.testLoss << ',' << e.testAccuracy << '\n';
        });
        std::printf("%.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage();
        return 2;
    }
    return 0;
}

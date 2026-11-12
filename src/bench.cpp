// Multiplexed networks vs MLPs with (about) the same number of parameters, several seeds.
//
//   mnn_bench [--data letter-recognition.data] [--epochs 30] [--seeds 3] [--out results]
// Writes <out>/benchmark.md (mean ± std of the final test accuracy) and <out>/curves.csv.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "mnn/data.hpp"
#include "mnn/model.hpp"

using namespace mnn;

struct Entry {
    std::string task, name;
    std::vector<int> main, control;
};

static std::string sizes(const std::vector<int> &v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) s += (i ? "-" : "") + std::to_string(v[i]);
    return s.empty() ? "–" : s;
}

int main(int argc, char **argv) {
    std::string data = "letter-recognition.data", outDir = "results";
    int epochs = 30, seeds = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--data") data = argv[i + 1];
        else if (a == "--epochs") epochs = std::stoi(argv[i + 1]);
        else if (a == "--seeds") seeds = std::stoi(argv[i + 1]);
        else if (a == "--out") outDir = argv[i + 1];
        else return std::fprintf(stderr, "unknown argument %s\n", a.c_str()), 2;
    }
    const std::vector<Entry> entries = {
        {"gated", "MLP", {16, 64, 4}, {}},
        {"gated", "MLP (wide)", {16, 503, 4}, {}},
        {"gated", "MLP (deep)", {16, 92, 92, 4}, {}},
        {"gated", "multiplexed (no hidden layer)", {16, 4}, {16}},
        {"gated", "multiplexed", {16, 32, 4}, {16, 8}},
        {"letter", "MLP", {16, 32, 26}, {}},
        {"letter", "MLP (wide)", {16, 397, 26}, {}},
        {"letter", "MLP (deep)", {16, 128, 96, 26}, {}},
        {"letter", "multiplexed (first layer)", {16, 32, 26}, {16}},
        {"letter", "multiplexed", {16, 32, 26}, {16, 8}},
    };
    std::filesystem::create_directories(outDir);
    std::ofstream curves(outDir + "/curves.csv"), md(outDir + "/benchmark.md");
    curves << "task,model,main,control,seed,epoch,train_loss,test_loss,test_accuracy\n";
    md << "| task | model | main | control | parameters | test accuracy (mean ± std, " << seeds
       << " seeds) | best epoch acc | s / run |\n|---|---|---|---|---|---|---|---|\n";

    Split gated = makeGated(8, 8, 4, 8000, 4000, 12345);
    Split letters;
    bool haveLetters = true;
    try {
        letters = loadLetters(data);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "skipping letter recognition: %s\n", e.what());
        haveLetters = false;
    }
    for (const auto &en : entries) {
        if (en.task == "letter" && !haveLetters) continue;
        const Split &s = en.task == "letter" ? letters : gated;
        std::vector<double> finals, bests;
        double seconds = 0;
        size_t nparams = 0;
        for (int seed = 1; seed <= seeds; seed++) {
            Config cfg;
            cfg.main = en.main, cfg.control = en.control, cfg.seed = (uint64_t)seed;
            Model m(cfg);
            nparams = m.parameterCount();
            Optimizer opt;
            opt.lr = 1e-3;
            double best = 0, last = 0;
            const auto t0 = std::chrono::steady_clock::now();
            train(m, opt, s, epochs, 32, (uint64_t)seed + 1000, [&](const EpochStats &e) {
                curves << en.task << ',' << en.name << ',' << sizes(en.main) << ',' << sizes(en.control) << ','
                       << seed << ',' << e.epoch << ',' << e.trainLoss << ',' << e.testLoss << ','
                       << e.testAccuracy << '\n';
                best = std::max(best, e.testAccuracy), last = e.testAccuracy;
            });
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            finals.push_back(last), bests.push_back(best);
            std::printf("%-6s %-30s seed %d: final %.4f best %.4f\n", en.task.c_str(), en.name.c_str(), seed, last,
                        best);
            std::fflush(stdout);
        }
        auto meanStd = [](const std::vector<double> &v) {
            double mu = 0, var = 0;
            for (double x : v) mu += x;
            mu /= (double)v.size();
            for (double x : v) var += (x - mu) * (x - mu);
            return std::make_pair(mu, v.size() > 1 ? std::sqrt(var / (double)(v.size() - 1)) : 0.0);
        };
        auto [mu, sd] = meanStd(finals);
        auto [bmu, bsd] = meanStd(bests);
        char row[512];
        std::snprintf(row, sizeof row, "| %s | %s | %s | %s | %zu | %.2f ± %.2f %% | %.2f %% | %.1f |\n",
                      en.task.c_str(), en.name.c_str(), sizes(en.main).c_str(), sizes(en.control).c_str(), nparams,
                      100 * mu, 100 * sd, 100 * bmu, seconds / seeds);
        md << row;
        md.flush();
        (void)bsd;
    }
    std::printf("wrote %s/benchmark.md and %s/curves.csv\n", outDir.c_str(), outDir.c_str());
    return 0;
}

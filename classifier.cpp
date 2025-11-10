// classifier.cpp
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "csvstream.hpp"
using namespace std;
#ifndef DEBUG_SCORES
#define DEBUG_SCORES 0
#endif

// Returns a set of unique whitespace-delimited words
static set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) words.insert(word);
  return words;
}

class NaiveBayesClassifier {
public:
  // Train from CSV with columns "tag" and "content".
  // If echo_training is true, print the "training data:" lines.
  void train(const string &train_filename, bool echo_training) {
    clear();
    if (echo_training) {
      cout << "training data:" << '\n';
    }
    try {
      csvstream csvin(train_filename);
      map<string, string> row;
      while (csvin >> row) {
        const string &label = row.at("tag");
        const string &content = row.at("content");
        if (echo_training) {
          cout << "  label = " << label << ", content = " << content << '\n';
        }
        ++num_posts_;
        ++label_counts_[label];
        // Unique words per post (Bernoulli presence features)
        set<string> words = unique_words(content);
        vocabulary_.insert(words.begin(), words.end());
        // Count presence per label and overall (per post)
        for (const string &w : words) {
          ++word_post_counts_[w];
          ++label_word_post_counts_[label][w];
        }
      }
    } catch (const csvstream_exception &e) {
      cout << e.what() << endl;
      throw;
    }
  }

  // Print the training summary.
  // - Train-only mode: prints vocabulary size, blank line, classes, parameters, blank line
  // - Test mode: prints only "trained on N examples" and a single blank line
  void print_train_summary(bool train_only_mode) const {
    cout.precision(3);
    cout << "trained on " << num_posts_ << " examples" << '\n';
    if (train_only_mode) {
      // Matches autograder's sequence exactly
      cout << "vocabulary size = " << vocabulary_.size() << '\n';
      cout << '\n';
      print_classes();
      print_classifier_parameters();
      cout << '\n';
    } else {
      cout << '\n';
    }
  }

  // Predict a label for content; returns {predicted_label, log_score}
  pair<string, double> predict(const string &content) const {
    set<string> words = unique_words(content);
    string best_label;
    double best_score = -numeric_limits<double>::infinity();

    for (const auto &kv : label_counts_) {
      const string &label = kv.first;
      double score = log_prior(label);

      // Sum contributions from each unique word (alphabetical via set)
      for (const string &w : words) {
        // Only apply Naive Bayes likelihoods if the word appeared in training.
        if (vocabulary_.count(w)) {
          const double ll = log_likelihood(label, w);
#if DEBUG_SCORES
          cerr << "[DBG] +" << ll << " from word '" << w
               << "' for label '" << label << "'\n";
#endif
          score += ll;
        }
      }

#if DEBUG_SCORES
      cerr << "[DBG] total score for label '" << label
           << "' = " << score << "\n";
#endif

      // Tie-breaker: alphabetical label if scores equal
      if (score > best_score ||
          (score == best_score && (best_label.empty() || label < best_label))) {
        best_score = score;
        best_label = label;
      }
    }

    return {best_label, best_score};
  }

private:
  // Learned state
  size_t num_posts_ = 0;
  set<string> vocabulary_;
  map<string, int> label_counts_;                        // label -> #posts
  map<string, int> word_post_counts_;                    // word -> #posts overall
  map<string, map<string, int>> label_word_post_counts_; // label -> (word -> #posts)

  void clear() {
    num_posts_ = 0;
    vocabulary_.clear();
    label_counts_.clear();
    word_post_counts_.clear();
    label_word_post_counts_.clear();
  }

  // log prior = log( count(label) / total_posts )
  double log_prior(const string &label) const {
    const double c = static_cast<double>(label_counts_.at(label));
    const double n = static_cast<double>(num_posts_);
    return std::log(c / n);
  }

  // EECS 280 likelihood rules (Bernoulli, presence-only) to match reference:
  //   seen with this label: log( count(label,word) / count(label) )
  //   seen somewhere else:  log( 1 / count(label) )
  //   never seen anywhere:  log( 1 / num_posts )
  double log_likelihood(const string &label, const string &word) const {
    const int label_count = label_counts_.at(label);
    const int num_posts_total = static_cast<int>(num_posts_);

    int label_word_count = 0;
    auto it_label = label_word_post_counts_.find(label);
    if (it_label != label_word_post_counts_.end()) {
      auto it_word = it_label->second.find(word);
      if (it_word != it_label->second.end()) {
        label_word_count = it_word->second;
      }
    }

    if (label_word_count > 0) {
      return std::log(static_cast<double>(label_word_count) /
                      static_cast<double>(label_count));
    } else if (word_post_counts_.count(word) > 0) {
      return std::log(1.0 / static_cast<double>(label_count));
    } else {
      return std::log(1.0 / static_cast<double>(num_posts_total));
    }
  }

  void print_classes() const {
    cout << "classes:" << '\n';
    for (const auto &kv : label_counts_) {
      const string &label = kv.first;
      int count = kv.second;
      cout << "  " << label << ", " << count
           << " examples, log-prior = " << log_prior(label) << '\n';
    }
  }

  void print_classifier_parameters() const {
    cout << "classifier parameters:" << '\n';
    for (const auto &lkv : label_word_post_counts_) {
      const string &label = lkv.first;
      for (const auto &wkv : lkv.second) {
        const string &word = wkv.first;
        int count = wkv.second;
        cout << "  " << label << ":" << word
             << ", count = " << count
             << ", log-likelihood = " << log_likelihood(label, word) << '\n';
      }
    }
  }
};

static int run_train_only(const string &train_file) {
  NaiveBayesClassifier clf;
  try {
    cout.precision(3);
    // Train-only mode: echo the training posts
    clf.train(train_file, /*echo_training=*/true);
    clf.print_train_summary(/*train_only_mode=*/true);
  } catch (...) {
    return 1;
  }
  return 0;
}

static int run_train_and_test(const string &train_file, const string &test_file) {
  NaiveBayesClassifier clf;
  try {
    cout.precision(3);
    // Test mode: do NOT echo training posts
    clf.train(train_file, /*echo_training=*/false);
    clf.print_train_summary(/*train_only_mode=*/false);
    cout << "test data:" << '\n';
    int correct = 0;
    int total = 0;
    try {
      csvstream csvin(test_file);
      map<string, string> row;
      while (csvin >> row) {
        const string &correct_label = row.at("tag");
        const string &content = row.at("content");
        auto [pred_label, log_score] = clf.predict(content);
        cout << "  correct = " << correct_label
             << ", predicted = " << pred_label
             << ", log-probability score = " << log_score << '\n';
        cout << "  content = " << content << "\n\n";
        ++total;
        if (pred_label == correct_label) ++correct;
      }
    } catch (const csvstream_exception &e) {
      cout << e.what() << endl;
      return 1;
    }
    cout << "performance: " << correct << " / " << total
         << " posts predicted correctly" << '\n';
  } catch (...) {
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 3) {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }
  const string train_file = argv[1];
  if (argc == 2) {
    return run_train_only(train_file);
  } else {
    const string test_file = argv[2];
    return run_train_and_test(train_file, test_file);
  }
}

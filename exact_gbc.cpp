// Single-file adaptation of Exact-GBC-CPP for the HEDGE/CentRA internal-node
// objective. Build: g++ -O2 -std=c++17 -fopenmp exact_gbc_internal.cpp -o exact_gbc_internal
// Unreachable ordered pairs contribute 0; the raw score counts both
// orientations of an undirected pair. This differs from NetworkX's
// endpoints=False, which excludes pairs with group endpoints.



#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace exact_gbc {

using NodeId = std::int64_t;

struct Edge {
    int to = -1;
    double weight = 1.0;
};

struct LoadOptions {
    bool directed = false;
    bool weighted = false;
    std::int64_t num_nodes = -1;
    std::string nodes_file;
};

/**
 * Simple graph stored as outgoing and incoming adjacency lists.
 *
 * External node labels may be arbitrary signed 64-bit integers.  Internally,
 * labels are remapped to the contiguous range [0, n), just as GEN-CIM does
 * before running its algorithms.
 */
class Graph {
public:
    static Graph load_edge_list(const std::string& path,
                                const LoadOptions& options);

    [[nodiscard]] std::size_t num_nodes() const noexcept { return labels_.size(); }
    [[nodiscard]] std::size_t num_edges() const noexcept { return num_edges_; }
    [[nodiscard]] bool directed() const noexcept { return directed_; }
    [[nodiscard]] bool weighted() const noexcept { return weighted_; }

    [[nodiscard]] const std::vector<Edge>& out_edges(int node) const {
        return out_.at(static_cast<std::size_t>(node));
    }
    [[nodiscard]] const std::vector<Edge>& in_edges(int node) const {
        return in_.at(static_cast<std::size_t>(node));
    }

    [[nodiscard]] int internal_id(NodeId label) const;
    [[nodiscard]] NodeId external_id(int node) const;

private:
    bool directed_ = false;
    bool weighted_ = false;
    std::size_t num_edges_ = 0;
    std::vector<NodeId> labels_;
    std::unordered_map<NodeId, int> label_to_id_;
    std::vector<std::vector<Edge>> out_;
    std::vector<std::vector<Edge>> in_;
};

struct Group {
    std::string name;
    std::vector<NodeId> nodes;
};

struct Result {
    std::string name;
    std::vector<NodeId> nodes;
    long double raw_gbc = 0.0L;
};

struct ComputeOptions {
    int threads = 1;
    bool verbose = true;
};

/**
 * Compute exact unnormalized ordered-pair internal-node GBC.
 *
 * Every ordered pair (s,t), s != t, contributes the fraction of its
 * shortest paths containing at least one INTERNAL node in the group.
 * Endpoints are allowed in the group but never cover a path.
 * Unreachable pairs contribute zero. This is not NetworkX endpoints=False.
 */
std::vector<Result> compute_exact_raw_gbc(const Graph& graph,
                                          const std::vector<Group>& groups,
                                          const ComputeOptions& options = {});

}  // namespace exact_gbc



#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace exact_gbc {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct RawEdge {
    NodeId source = 0;
    NodeId target = 0;
    double weight = 1.0;
};

struct InternalEdge {
    int source = -1;
    int target = -1;
    double weight = 1.0;
};

std::string trim_copy(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool is_comment_or_empty(const std::string& line) {
    const std::string clean = trim_copy(line);
    return clean.empty() || clean.front() == '#' || clean.front() == '%';
}

/** Workspace reused for each exact single-source shortest-path traversal. */
struct SsspWorkspace {
    explicit SsspWorkspace(std::size_t n)
        : distance(n, kInfinity), sigma(n, 0.0), delta(n, 0.0) {
        order.reserve(n);
    }

    void reset() {
        std::fill(distance.begin(), distance.end(), kInfinity);
        std::fill(sigma.begin(), sigma.end(), 0.0);
        std::fill(delta.begin(), delta.end(), 0.0);
        order.clear();
    }

    std::vector<double> distance;
    std::vector<long double> sigma;
    std::vector<double> delta;
    std::vector<int> order;
};

void run_unweighted_sssp(const Graph& graph, int source, SsspWorkspace& ws) {
    ws.reset();
    std::deque<int> queue;
    ws.distance[static_cast<std::size_t>(source)] = 0.0;
    ws.sigma[static_cast<std::size_t>(source)] = 1.0;
    queue.push_back(source);

    while (!queue.empty()) {
        const int v = queue.front();
        queue.pop_front();
        ws.order.push_back(v);

        const double next_distance = ws.distance[static_cast<std::size_t>(v)] + 1.0;
        for (const Edge& edge : graph.out_edges(v)) {
            const int w = edge.to;
            if (!std::isfinite(ws.distance[static_cast<std::size_t>(w)])) {
                ws.distance[static_cast<std::size_t>(w)] = next_distance;
                queue.push_back(w);
            }
            if (ws.distance[static_cast<std::size_t>(w)] == next_distance) {
                ws.sigma[static_cast<std::size_t>(w)] +=
                    ws.sigma[static_cast<std::size_t>(v)];
            }
        }
    }
}

void run_weighted_sssp(const Graph& graph, int source, SsspWorkspace& ws) {
    ws.reset();
    using QueueItem = std::pair<double, int>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> queue;

    ws.distance[static_cast<std::size_t>(source)] = 0.0;
    ws.sigma[static_cast<std::size_t>(source)] = 1.0;
    queue.emplace(0.0, source);

    while (!queue.empty()) {
        const auto [distance_v, v] = queue.top();
        queue.pop();
        if (distance_v != ws.distance[static_cast<std::size_t>(v)]) {
            continue;
        }
        ws.order.push_back(v);

        for (const Edge& edge : graph.out_edges(v)) {
            const int w = edge.to;
            const double candidate = distance_v + edge.weight;
            double& distance_w = ws.distance[static_cast<std::size_t>(w)];
            long double& sigma_w = ws.sigma[static_cast<std::size_t>(w)];
            if (candidate < distance_w) {
                distance_w = candidate;
                sigma_w = ws.sigma[static_cast<std::size_t>(v)];
                queue.emplace(candidate, w);
            } else if (candidate == distance_w) {
                // Positive weights guarantee that every predecessor is settled
                // before w; equal shortest paths can therefore be accumulated.
                sigma_w += ws.sigma[static_cast<std::size_t>(v)];
            }
        }
    }
}

void run_sssp(const Graph& graph, int source, SsspWorkspace& ws) {
    if (graph.weighted()) {
        run_weighted_sssp(graph, source, ws);
    } else {
        run_unweighted_sssp(graph, source, ws);
    }
}

}  // namespace

Graph Graph::load_edge_list(const std::string& path, const LoadOptions& options) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open graph file: " + path);
    }

    std::vector<RawEdge> raw_edges;
    std::vector<NodeId> labels;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (is_comment_or_empty(line)) {
            continue;
        }
        std::istringstream row(line);
        RawEdge edge;
        if (!(row >> edge.source >> edge.target)) {
            throw std::runtime_error("Invalid edge at " + path + ":" +
                                     std::to_string(line_number));
        }
        if (options.weighted && !(row >> edge.weight)) {
            throw std::runtime_error("Missing edge weight at " + path + ":" +
                                     std::to_string(line_number));
        }
        if (!std::isfinite(edge.weight) || edge.weight <= 0.0) {
            throw std::runtime_error(
                "Exact weighted GBC requires every edge weight to be finite and > 0; "
                "invalid value at line " + std::to_string(line_number));
        }
        raw_edges.push_back(edge);
        labels.push_back(edge.source);
        labels.push_back(edge.target);
    }

    if (options.num_nodes >= 0) {
        for (NodeId node = 0; node < options.num_nodes; ++node) {
            labels.push_back(node);
        }
    }
    if (!options.nodes_file.empty()) {
        std::ifstream nodes_input(options.nodes_file);
        if (!nodes_input) {
            throw std::runtime_error("Cannot open nodes file: " + options.nodes_file);
        }
        line_number = 0;
        while (std::getline(nodes_input, line)) {
            ++line_number;
            if (is_comment_or_empty(line)) {
                continue;
            }
            std::istringstream row(line);
            NodeId node = 0;
            if (!(row >> node)) {
                throw std::runtime_error("Invalid node label at " + options.nodes_file +
                                         ":" + std::to_string(line_number));
            }
            labels.push_back(node);
        }
    }

    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (labels.empty()) {
        throw std::runtime_error("The graph has no nodes.");
    }

    Graph graph;
    graph.directed_ = options.directed;
    graph.weighted_ = options.weighted;
    graph.labels_ = labels;
    graph.label_to_id_.reserve(labels.size() * 2);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        graph.label_to_id_.emplace(labels[i], static_cast<int>(i));
    }

    std::vector<InternalEdge> edges;
    edges.reserve(raw_edges.size());
    for (const RawEdge& raw : raw_edges) {
        int u = graph.label_to_id_.at(raw.source);
        int v = graph.label_to_id_.at(raw.target);
        if (u == v) {
            continue;  // GEN-CIM preprocessing also removes self-loops.
        }
        if (!options.directed && v < u) {
            std::swap(u, v);
        }
        edges.push_back({u, v, options.weighted ? raw.weight : 1.0});
    }

    std::sort(edges.begin(), edges.end(), [](const InternalEdge& lhs,
                                              const InternalEdge& rhs) {
        return std::tie(lhs.source, lhs.target, lhs.weight) <
               std::tie(rhs.source, rhs.target, rhs.weight);
    });

    // Convert a possibly noisy edge list into a simple graph.  If a weighted
    // pair occurs more than once, retain its minimum distance.
    std::vector<InternalEdge> unique_edges;
    unique_edges.reserve(edges.size());
    for (const InternalEdge& edge : edges) {
        if (!unique_edges.empty() && unique_edges.back().source == edge.source &&
            unique_edges.back().target == edge.target) {
            unique_edges.back().weight = std::min(unique_edges.back().weight, edge.weight);
        } else {
            unique_edges.push_back(edge);
        }
    }

    graph.num_edges_ = unique_edges.size();
    graph.out_.assign(labels.size(), {});
    graph.in_.assign(labels.size(), {});
    for (const InternalEdge& edge : unique_edges) {
        graph.out_[static_cast<std::size_t>(edge.source)].push_back(
            {edge.target, edge.weight});
        graph.in_[static_cast<std::size_t>(edge.target)].push_back(
            {edge.source, edge.weight});
        if (!options.directed) {
            graph.out_[static_cast<std::size_t>(edge.target)].push_back(
                {edge.source, edge.weight});
            graph.in_[static_cast<std::size_t>(edge.source)].push_back(
                {edge.target, edge.weight});
        }
    }

    for (auto& adjacency : graph.out_) {
        std::sort(adjacency.begin(), adjacency.end(),
                  [](const Edge& a, const Edge& b) { return a.to < b.to; });
    }
    for (auto& adjacency : graph.in_) {
        std::sort(adjacency.begin(), adjacency.end(),
                  [](const Edge& a, const Edge& b) { return a.to < b.to; });
    }
    return graph;
}

int Graph::internal_id(NodeId label) const {
    const auto it = label_to_id_.find(label);
    if (it == label_to_id_.end()) {
        throw std::out_of_range("Node " + std::to_string(label) +
                                " is not present in the graph.");
    }
    return it->second;
}

NodeId Graph::external_id(int node) const {
    return labels_.at(static_cast<std::size_t>(node));
}

std::vector<Result> compute_exact_raw_gbc(const Graph& graph,
                                          const std::vector<Group>& groups,
                                          const ComputeOptions& options) {
    // Source code adapted from Exact-GBC-CPP. SSSP and graph parsing are
    // retained; scoring uses a shortest-path DAG dynamic program because the
    // original endpoint correction excluded entire pairs with s or t in C.
    const std::size_t n = graph.num_nodes();
    if (n < 2 || groups.empty()) {
        throw std::invalid_argument("Expected at least two nodes and one group.");
    }
    std::vector<std::vector<unsigned char>> in_group;
    std::vector<Result> results;
    in_group.reserve(groups.size());
    results.reserve(groups.size());
    for (const Group& g : groups) {
        std::vector<unsigned char> mark(n, 0);
        Result result;
        result.name = g.name;
        for (NodeId label : g.nodes) {
            const int id = graph.internal_id(label);
            if (!mark[static_cast<std::size_t>(id)]) {
                mark[static_cast<std::size_t>(id)] = 1;
                result.nodes.push_back(label);
            }
        }
        if (result.nodes.empty() || result.nodes.size() > n) {
            throw std::invalid_argument("Group must be nonempty and contained in V.");
        }
        std::sort(result.nodes.begin(), result.nodes.end());
        in_group.push_back(std::move(mark));
        results.push_back(std::move(result));
    }

    int thread_count = std::max(1, options.threads);
#ifdef _OPENMP
    thread_count = std::min(thread_count, omp_get_max_threads());
#else
    thread_count = 1;
#endif
    std::vector<std::vector<long double>> partial(
        static_cast<std::size_t>(thread_count),
        std::vector<long double>(groups.size(), 0.0L));
    std::atomic<std::size_t> done{0};
    if (options.verbose) {
        std::cerr << "[INFO] Exact internal-node GBC: |V|=" << n
                  << ", |E|=" << graph.num_edges()
                  << ", groups=" << groups.size()
                  << ", threads=" << thread_count << '\n';
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count) if(thread_count > 1)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        SsspWorkspace ws(n);
        std::vector<long double> avoid(n, 0.0L);
        auto& scores = partial[static_cast<std::size_t>(tid)];
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::int64_t source_i = 0;
             source_i < static_cast<std::int64_t>(n); ++source_i) {
            const int source = static_cast<int>(source_i);
            run_sssp(graph, source, ws);
            // For each candidate group, avoid[t] counts shortest s-t paths
            // whose INTERNAL vertices avoid C. Even when s or t is in C,
            // its endpoint membership does not block a path. A group node v
            // contributes as a target, but cannot extend an avoiding path.
            for (std::size_t gi = 0; gi < groups.size(); ++gi) {
                std::fill(avoid.begin(), avoid.end(), 0.0L);
                avoid[static_cast<std::size_t>(source)] = 1.0L;
                for (const int v : ws.order) {
                    const std::size_t vi = static_cast<std::size_t>(v);
                    if (avoid[vi] == 0.0L ||
                        (v != source && in_group[gi][vi])) continue;
                    for (const Edge& edge : graph.out_edges(v)) {
                        const std::size_t wi = static_cast<std::size_t>(edge.to);
                        const double step = graph.weighted() ? edge.weight : 1.0;
                        if (ws.distance[vi] + step == ws.distance[wi]) {
                            avoid[wi] += avoid[vi];
                        }
                    }
                }
                for (const int target : ws.order) {
                    if (target == source) continue;
                    const std::size_t ti = static_cast<std::size_t>(target);
                    const long double denominator = ws.sigma[ti];
                    long double fraction = 1.0L - avoid[ti] / denominator;
                    // Guard only roundoff; both counters refer to the same DAG.
                    if (fraction < 0.0L && fraction > -1e-12L) fraction = 0.0L;
                    if (fraction > 1.0L && fraction < 1.0L + 1e-12L) fraction = 1.0L;
                    scores[gi] += fraction;
                }
            }
            const auto count = ++done;
            if (options.verbose && (count == n || count % std::max<std::size_t>(1, n / 10) == 0)) {
#ifdef _OPENMP
#pragma omp critical(exact_internal_progress)
#endif
                std::cerr << "\r[INFO] Sources processed: " << 100 * count / n << "%" << std::flush;
            }
        }
    }
    if (options.verbose) std::cerr << '\n';
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
        for (int tid = 0; tid < thread_count; ++tid) {
            results[gi].raw_gbc += partial[static_cast<std::size_t>(tid)][gi];
        }
    }
    return results;
}

}  // namespace exact_gbc



#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using exact_gbc::Group;
using exact_gbc::NodeId;

struct CliOptions {
    std::string graph_path;
    std::vector<std::string> inline_groups;
    std::string groups_file;
    std::vector<std::string> json_files;
    std::string json_key = "seed_set";
    std::string output_path;
    std::string nodes_file;
    std::int64_t num_nodes = -1;
    int threads = std::max(1U, std::thread::hardware_concurrency());
    bool directed = false;
    bool weighted = false;
    bool groups_are_internal = false;
    bool quiet = false;
};

std::string trim_copy(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

void print_usage(std::ostream& out) {
    out << R"USAGE(
Exact Group Betweenness Centrality: internal nodes, ordered pairs

Usage:
  exact_gbc --graph EDGE_LIST --group "name: node1,node2,..." [options]
  exact_gbc --graph EDGE_LIST --groups-file GROUPS.txt [options]
  exact_gbc --graph EDGE_LIST --group-json result.json [options]

Required:
  --graph PATH            SNAP-style edge list: "u v" or "u v weight".

Group input (one or more may be combined):
  --group SPEC            One group. Repeatable. Name is optional.
  --groups-file PATH      One group per line: "method_name: n1 n2 ...".
  --group-json PATH       Read an integer array from a GEN-CIM result JSON.
                         Repeatable.
  --json-key KEY          JSON array key (default: seed_set).
  --group-ids SPACE       "external" (default) or "internal".  GEN-CIM saved
                         seed sets normally use internal contiguous IDs.

Graph options:
  --directed              Treat every input edge as directed.
  --weighted              Read positive distance from the third column.
  --num-nodes N           Also include isolated labels 0, ..., N-1.
  --nodes-file PATH       Also include node labels listed in PATH.

Runtime/output:
  --threads N             OpenMP CPU threads (default: hardware threads).
  --output PATH           Also save the JSON result to PATH.
  --quiet                 Hide progress messages (JSON still goes to stdout).
  --help                  Show this message.

B(C) = sum over s != t with a path of sigma_st_internal(C) / sigma_st.
Pairs with s or t in C ARE included, but only internal nodes count.
For undirected graphs, this is twice the unordered-pair raw score.
Normalized HEDGE/CentRA score = raw_gbc / (n*(n-1)).
)USAGE";
}

std::string require_value(int& index, int argc, char** argv,
                          const std::string& option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value after " + option);
    }
    return argv[++index];
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--graph") {
            options.graph_path = require_value(i, argc, argv, arg);
        } else if (arg == "--group") {
            options.inline_groups.push_back(require_value(i, argc, argv, arg));
        } else if (arg == "--groups-file") {
            options.groups_file = require_value(i, argc, argv, arg);
        } else if (arg == "--group-json") {
            options.json_files.push_back(require_value(i, argc, argv, arg));
        } else if (arg == "--json-key") {
            options.json_key = require_value(i, argc, argv, arg);
        } else if (arg == "--group-ids") {
            const std::string space = require_value(i, argc, argv, arg);
            if (space == "internal") {
                options.groups_are_internal = true;
            } else if (space == "external") {
                options.groups_are_internal = false;
            } else {
                throw std::invalid_argument(
                    "--group-ids must be either 'internal' or 'external'.");
            }
        } else if (arg == "--output") {
            options.output_path = require_value(i, argc, argv, arg);
        } else if (arg == "--nodes-file") {
            options.nodes_file = require_value(i, argc, argv, arg);
        } else if (arg == "--num-nodes") {
            options.num_nodes = std::stoll(require_value(i, argc, argv, arg));
            if (options.num_nodes < 0) {
                throw std::invalid_argument("--num-nodes must be non-negative.");
            }
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value(i, argc, argv, arg));
            if (options.threads < 1) {
                throw std::invalid_argument("--threads must be at least 1.");
            }
        } else if (arg == "--directed") {
            options.directed = true;
        } else if (arg == "--weighted") {
            options.weighted = true;
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(std::cout);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.graph_path.empty()) {
        throw std::invalid_argument("--graph is required.");
    }
    if (options.inline_groups.empty() && options.groups_file.empty() &&
        options.json_files.empty()) {
        throw std::invalid_argument(
            "Provide --group, --groups-file, or --group-json.");
    }
    return options;
}

std::vector<NodeId> parse_node_list(std::string text) {
    for (char& ch : text) {
        if (ch == ',' || ch == ';' || ch == '[' || ch == ']') {
            ch = ' ';
        }
    }
    std::istringstream input(text);
    std::vector<NodeId> nodes;
    NodeId node = 0;
    while (input >> node) {
        nodes.push_back(node);
    }
    if (nodes.empty()) {
        throw std::invalid_argument("A group contains no valid integer node labels.");
    }
    return nodes;
}

Group parse_group_spec(const std::string& spec, std::size_t sequence) {
    const auto colon = spec.find(':');
    Group group;
    if (colon == std::string::npos) {
        group.name = "group_" + std::to_string(sequence);
        group.nodes = parse_node_list(spec);
    } else {
        group.name = trim_copy(spec.substr(0, colon));
        group.nodes = parse_node_list(spec.substr(colon + 1));
        if (group.name.empty()) {
            group.name = "group_" + std::to_string(sequence);
        }
    }
    return group;
}

void append_groups_file(const std::string& path, std::vector<Group>& groups) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open groups file: " + path);
    }
    std::string line;
    while (std::getline(input, line)) {
        const std::string clean = trim_copy(line);
        if (clean.empty() || clean.front() == '#' || clean.front() == '%') {
            continue;
        }
        groups.push_back(parse_group_spec(clean, groups.size() + 1));
    }
}

std::string read_all(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

Group parse_json_group(const std::string& path, const std::string& key) {
    const std::string text = read_all(path);
    const std::string quoted_key = "\"" + key + "\"";
    const auto key_position = text.find(quoted_key);
    if (key_position == std::string::npos) {
        throw std::runtime_error("JSON key '" + key + "' not found in " + path);
    }
    const auto array_begin = text.find('[', key_position + quoted_key.size());
    if (array_begin == std::string::npos) {
        throw std::runtime_error("JSON key '" + key + "' is not followed by an array.");
    }
    const auto array_end = text.find(']', array_begin + 1);
    if (array_end == std::string::npos) {
        throw std::runtime_error("Unclosed JSON array for key '" + key + "'.");
    }

    Group group;
    group.name = std::filesystem::path(path).stem().string();
    group.nodes = parse_node_list(text.substr(array_begin, array_end - array_begin + 1));
    return group;
}

std::string escape_json(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(ch)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

std::string results_as_json(const CliOptions& options,
                            const exact_gbc::Graph& graph,
                            const std::vector<exact_gbc::Result>& results) {
    std::ostringstream output;
    output << std::setprecision(18);
    output << "{\n"
           << "  \"graph\": \"" << escape_json(options.graph_path) << "\",\n"
           << "  \"directed\": " << (graph.directed() ? "true" : "false") << ",\n"
           << "  \"weighted\": " << (graph.weighted() ? "true" : "false") << ",\n"
           << "  \"normalized\": false,\n"
           << "  \"endpoints_counted\": false,\n"
           << "  \"pairs_with_group_endpoints_included\": true,\n"
           << "  \"pair_domain\": \"all_distinct_ordered_pairs\",\n"
           << "  \"coverage\": \"at_least_one_internal_group_node\",\n"
           << "  \"input_group_id_space\": \""
           << (options.groups_are_internal ? "internal" : "external") << "\",\n"
           << "  \"num_nodes\": " << graph.num_nodes() << ",\n"
           << "  \"num_edges\": " << graph.num_edges() << ",\n"
           << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        output << "    {\n"
               << "      \"method\": \"" << escape_json(result.name) << "\",\n"
               << "      \"k\": " << result.nodes.size() << ",\n"
               << "      \"nodes\": [";
        for (std::size_t j = 0; j < result.nodes.size(); ++j) {
            if (j != 0) {
                output << ", ";
            }
            output << result.nodes[j];
        }
        output << "],\n"
               << "      \"raw_gbc\": " << result.raw_gbc << ",\n"
               << "      \"normalized_gbc\": "
               << result.raw_gbc /
                  (static_cast<long double>(graph.num_nodes()) * (graph.num_nodes() - 1))
               << "\n"
               << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);

        std::vector<Group> groups;
        for (const std::string& spec : options.inline_groups) {
            groups.push_back(parse_group_spec(spec, groups.size() + 1));
        }
        if (!options.groups_file.empty()) {
            append_groups_file(options.groups_file, groups);
        }
        for (const std::string& json_path : options.json_files) {
            groups.push_back(parse_json_group(json_path, options.json_key));
        }
        if (groups.empty()) {
            throw std::invalid_argument("No non-empty groups were loaded.");
        }

        exact_gbc::LoadOptions load_options;
        load_options.directed = options.directed;
        load_options.weighted = options.weighted;
        load_options.num_nodes = options.num_nodes;
        load_options.nodes_file = options.nodes_file;

        if (!options.quiet) {
            std::cerr << "[INFO] Loading graph: " << options.graph_path << '\n';
        }
        const exact_gbc::Graph graph =
            exact_gbc::Graph::load_edge_list(options.graph_path, load_options);

        if (options.groups_are_internal) {
            // GEN-CIM stores seed sets after remapping the graph to [0, n).
            // Convert those IDs back to external labels expected by the public
            // evaluator API.  For the same preprocessed SNAP edge list,
            // Graph::external_id follows GEN-CIM's sorted-label remapping.
            for (Group& group : groups) {
                for (NodeId& node : group.nodes) {
                    if (node < 0 || static_cast<std::size_t>(node) >= graph.num_nodes()) {
                        throw std::out_of_range(
                            "Internal group node " + std::to_string(node) +
                            " is outside [0, |V|)." );
                    }
                    node = graph.external_id(static_cast<int>(node));
                }
            }
        }

        exact_gbc::ComputeOptions compute_options;
        compute_options.threads = options.threads;
        compute_options.verbose = !options.quiet;
        const auto results =
            exact_gbc::compute_exact_raw_gbc(graph, groups, compute_options);

        const std::string json = results_as_json(options, graph, results);
        std::cout << json;
        if (!options.output_path.empty()) {
            std::ofstream output(options.output_path);
            if (!output) {
                throw std::runtime_error("Cannot write output file: " +
                                         options.output_path);
            }
            output << json;
            if (!options.quiet) {
                std::cerr << "[INFO] Saved exact GBC results: "
                          << options.output_path << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << "\n\n";
        print_usage(std::cerr);
        return 1;
    }
}

// AdaAlg (paper Algorithm 1) adapted to the internal-node-only GBC objective.
// Two persistent independent sample pools are extended each round. The
// candidate is selected from S, validated on T, and the final group is
// selected from S union T as in the paper. This implementation makes no unproved claim
// that the original theorem transfers unchanged to the adapted objective.
// If its intrinsic Qmax rounds end before the stopping condition holds,
// the returned union-greedy group is reported as uncertified.
// Build: g++ -O2 -std=c++17 -Wall -Wextra adaalg.cpp -o adaalg
// Input: unweighted simple edge list with integer vertex labels.
// Directed graphs require --directed; otherwise edges are undirected.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using Label = std::int64_t;
struct Graph {
    std::vector<Label> label;
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<int>> incoming;
    std::size_t edges = 0;
};
struct Options {
    std::string graph, output;
    bool directed = false;
    int k = -1;
    double epsilon = 0.1, delta = 0.05;
    std::uint64_t seed = 1;
};

void usage() {
    std::cerr << "Usage: adaalg --graph EDGE_LIST --k K "
                 "[--directed] [--epsilon 0.1] [--delta 0.05] "
                 "[--seed 1] [--output result.json]\n"
                 "The estimated_gbc is union-sample coverage. "
                 "Both independent pools grow until the paper stopping condition "
                 "or its intrinsic Qmax rounds.\n";
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") { usage(); std::exit(0); }
        if (key == "--directed") { o.directed = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("Missing argument for " + key);
        const std::string val = argv[++i];
        if (key == "--graph") o.graph = val;
        else if (key == "--k") o.k = std::stoi(val);
        else if (key == "--epsilon") o.epsilon = std::stod(val);
        else if (key == "--delta") o.delta = std::stod(val);
        else if (key == "--max-samples")
            throw std::invalid_argument("AdaAlg has no fixed sample cap; remove --max-samples");
        else if (key == "--seed") o.seed = std::stoull(val);
        else if (key == "--output") o.output = val;
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (o.graph.empty() || o.k <= 0 || !(o.epsilon > 0 && o.epsilon < 0.632) ||
        !(o.delta > 0 && o.delta < 1))
        throw std::invalid_argument("Invalid --graph, --k, --epsilon, or --delta");
    return o;
}
Graph load_graph(const std::string& file, bool directed) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("Cannot open graph: " + file);
    std::vector<std::pair<Label,Label>> edges;
    std::vector<Label> labels;
    std::string line;
    while (std::getline(in,line)) {
        auto p=line.find_first_not_of(" \t\r\n");
        if (p==std::string::npos || line[p]=='#' || line[p]=='%') continue;
        std::istringstream row(line);
        Label a,b;
        if (!(row>>a>>b)) throw std::runtime_error("Expected integer edge: " + line);
        if (a==b) continue;
        if (directed) edges.emplace_back(a,b);
        else edges.emplace_back(std::min(a,b),std::max(a,b));
        labels.push_back(a); labels.push_back(b);
    }
    std::sort(labels.begin(),labels.end());
    labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
    if (labels.size() < 2) throw std::runtime_error("Need at least two nodes");
    std::sort(edges.begin(),edges.end());
    edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    Graph g; g.label=std::move(labels); g.edges=edges.size();
    g.adj.resize(g.label.size()); g.incoming.resize(g.label.size());
    for (const auto& [a,b]:edges) {
        int u=static_cast<int>(std::lower_bound(g.label.begin(),g.label.end(),a)-g.label.begin());
        int v=static_cast<int>(std::lower_bound(g.label.begin(),g.label.end(),b)-g.label.begin());
        g.adj[u].push_back(v);g.incoming[v].push_back(u);
        if (!directed) {g.adj[v].push_back(u);g.incoming[u].push_back(v);}
    }
    for (auto& ns:g.adj) std::sort(ns.begin(),ns.end());
    for (auto& ns:g.incoming) std::sort(ns.begin(),ns.end());
    return g;
}

using Sample=std::vector<int>;
// One uniform ordered pair s!=t, followed by one uniformly chosen shortest
// s-t path. The reverse predecessor walk chooses with probability sigma[p]/sigma[v].
Sample sample_path(const Graph& g, std::mt19937_64& rng) {
    const int n=static_cast<int>(g.label.size());
    std::uniform_int_distribution<int> vertex(0,n-1);
    int s=vertex(rng),t=vertex(rng);
    while (t==s) t=vertex(rng);
    std::vector<int> distance(n,-1),queue(n);
    std::vector<long double> count(n,0.0L);
    int begin=0,end=0;
    queue[end++]=s;distance[s]=0;count[s]=1;
    while (begin<end) {
        int v=queue[begin++];
        if (distance[t]>=0 && distance[v]>=distance[t]) break;
        for (int w:g.adj[v]) {
            if (distance[w]<0) { distance[w]=distance[v]+1; queue[end++]=w; }
            if (distance[w]==distance[v]+1) count[w]+=count[v];
        }
    }
    Sample inside;
    if (distance[t]<0) return inside; // unreachable pair: zero coverage
    for (int v=t;v!=s;) {
        long double total=0;
        for (int w:g.incoming[v]) if (distance[w]==distance[v]-1) total+=count[w];
        std::uniform_real_distribution<long double> choose(0.0L,total);
        long double x=choose(rng);
        int predecessor=-1;
        for (int w:g.incoming[v]) if (distance[w]==distance[v]-1) {
            predecessor=w; x-=count[w]; if (x<=0) break;
        }
        if (predecessor<0) throw std::runtime_error("Invalid shortest-path DAG");
        v=predecessor;
        if (v!=s) inside.push_back(v); // t and s never enter the hyperedge
    }
    return inside;
}

std::vector<Sample> draw(const Graph& g,std::size_t count,std::mt19937_64& rng) {
    std::vector<Sample> samples;
    samples.reserve(count);
    for (std::size_t i=0;i<count;++i) samples.push_back(sample_path(g,rng));
    return samples;
}
std::vector<int> greedy(const std::vector<Sample>& samples,int n,int k,std::size_t& covered_count) {
    std::vector<std::vector<int>> incident(n);
    for (std::size_t j=0;j<samples.size();++j)
        for (int v:samples[j]) incident[v].push_back(static_cast<int>(j));
    std::vector<int> gain(n),selected;
    for (int v=0;v<n;++v) gain[v]=static_cast<int>(incident[v].size());
    std::vector<unsigned char> covered(samples.size(),0),picked(n,0);
    covered_count=0;
    for (int step=0;step<k;++step) {
        int best=-1;
        for (int v=0;v<n;++v) if (!picked[v] && (best<0 || gain[v]>gain[best])) best=v;
        if (best<0) break;
        picked[best]=1;selected.push_back(best);
        for (int sample_id:incident[best]) if (!covered[sample_id]) {
            covered[sample_id]=1;++covered_count;
            for (int v:samples[sample_id]) --gain[v];
        }
    }
    return selected;
}
double evaluate(const std::vector<Sample>& samples,const std::vector<int>& chosen,int n) {
    std::vector<unsigned char> selected(n,0);
    for (int v:chosen) selected[v]=1;
    std::size_t covered=0;
    for (const Sample& path:samples) {
        for (int v:path) if (selected[v]) { ++covered;break; }
    }
    return static_cast<double>(covered)/static_cast<double>(samples.size());
}

int main(int argc,char** argv) {
    try {
        const Options opts=parse(argc,argv);
        const Graph g=load_graph(opts.graph, opts.directed);
        const int n=static_cast<int>(g.label.size());
        if (opts.k>=n) throw std::invalid_argument("Require k < |V|");
        // Two independent streams for the persistent training/validation pools.
        std::mt19937_64 train_rng(opts.seed), validate_rng(opts.seed^0x9e3779b97f4a7c15ULL);
        constexpr double e=2.71828182845904523536;
        const double alpha=opts.epsilon/(2.0-1.0/e);
        // AdaAlg paper Algorithm 1, Eq. (12): c2=(0.8+3 epsilon)/alpha^2.
        // The distributed source uses (2+alpha)/alpha^2 here; we follow the paper.
        const double c2=(0.8+3.0*opts.epsilon)/(alpha*alpha);
        const double b=std::max(1.1,(3*c2+2+std::sqrt(18*c2+4))/(3*c2-2));
        const int qmax=std::max(1,static_cast<int>(std::ceil(std::log(double(n)*(n-1))/std::log(b))));
        const double theta=std::log(4.0/opts.delta)*c2;
        double target=theta,guess_normalized=1.0;
        int successes=0,round=0;
        double train_score=0, validation_score=0,epsilon_sum=std::numeric_limits<double>::infinity();
        std::string status="schedule_exhausted";
        std::vector<Sample> training, validation;
        std::vector<int> selected;
        std::size_t per_pool=0;
        for (int q=1;q<=qmax;++q) {
            guess_normalized/=b;target*=b;
            if(!std::isfinite(target)||target>static_cast<double>(std::numeric_limits<int>::max()/2))
                throw std::overflow_error("AdaAlg path count exceeds indexed hyperedge capacity");
            const std::size_t count=static_cast<std::size_t>(std::ceil(target));
            if (count<=per_pool) break;
            auto new_training=draw(g,count-per_pool,train_rng);
            auto new_validation=draw(g,count-per_pool,validate_rng);
            training.insert(training.end(),std::make_move_iterator(new_training.begin()),
                            std::make_move_iterator(new_training.end()));
            validation.insert(validation.end(),std::make_move_iterator(new_validation.begin()),
                              std::make_move_iterator(new_validation.end()));
            per_pool=count;
            std::size_t hits=0;
            auto candidate=greedy(training,n,opts.k,hits);
            const double observed=evaluate(validation,candidate,n);
            selected=std::move(candidate);
            round=q;
            train_score=static_cast<double>(hits)/count;
            validation_score=observed;
            if (observed>=guess_normalized) ++successes;
            if (successes>=1 && train_score>0.0) {
                // AdaAlg Eq. (9), including the factor 2 in its denominator.
                const double c1=std::log(4.0/opts.delta)/
                    (2.0*theta*std::pow(b,successes-2));
                const double epsilon1=(2*c1/3+std::sqrt(4*c1*c1/9+8*c1))/2;
                const double beta=1.0-observed/train_score;
                epsilon_sum=beta*(1-1/e)*(1-epsilon1)+(2-1/e)*epsilon1;
                if (epsilon_sum<=opts.epsilon) {status="stopping_condition_met";break;}
            }
        }
        const auto tentative=selected;
        // Paper Algorithm 1 lines 23-24 selects on S union T when certified.
        // Apply the same final selector to an exhausted schedule, without a claim
        // of the paper's approximation guarantee.
        training.insert(training.end(),std::make_move_iterator(validation.begin()),
                        std::make_move_iterator(validation.end()));
        std::size_t union_hits=0;
        selected=greedy(training,n,opts.k,union_hits);
        const double union_score=static_cast<double>(union_hits)/training.size();
        std::ostringstream json;json<<std::setprecision(17);
        json<<"{\n  \"method\": \"AdaAlg_internal\",\n  \"graph\": \""<<opts.graph
            <<"\",\n  \"directed\": "<<(opts.directed ? "true" : "false")
            <<",\n  \"n\": "<<n<<",\n  \"m\": "<<g.edges
            <<",\n  \"k\": "<<opts.k<<",\n  \"nodes\": [";
        for (std::size_t i=0;i<selected.size();++i) {
            if (i) json<<", ";
            json<<g.label[selected[i]];
        }
        json<<"],\n  \"tentative_training_nodes\": [";
        for (std::size_t i=0;i<tentative.size();++i) {
            if (i) json<<", ";
            json<<g.label[tentative[i]];
        }
        json<<"],\n  \"estimated_gbc\": "<<union_score
            <<",\n  \"estimated_gbc_kind\": \"biased_union_coverage\""
            <<",\n  \"training_gbc_tentative\": "<<train_score
            <<",\n  \"validation_gbc_tentative\": "<<validation_score
            <<",\n  \"status\": \""<<status<<"\",\n  \"round\": "<<round
            <<",\n  \"samples_per_pool_last_round\": "<<per_pool
            <<",\n  \"samples_training\": "<<per_pool
            <<",\n  \"samples_validation\": "<<per_pool
            <<",\n  \"samples_total\": "<<training.size()
            <<",\n  \"sample_rule\": \"paper_adaptive_schedule_until_stopping_or_qmax\""
            <<",\n  \"epsilon\": "<<opts.epsilon<<",\n  \"delta\": "<<opts.delta
            <<",\n  \"seed\": "<<opts.seed
            <<",\n  \"paper_base_b\": "<<b
            <<",\n  \"paper_theta\": "<<theta
            <<",\n  \"paper_qmax\": "<<qmax;
        if (std::isfinite(epsilon_sum)) json<<",\n  \"stopping_expression\": "<<epsilon_sum;
        json<<"\n}\n";
        std::cout<<json.str();
        if (!opts.output.empty()) {
            std::ofstream out(opts.output);
            if (!out) throw std::runtime_error("Cannot write output: "+opts.output);
            out<<json.str();
        }
        return 0;
    } catch(const std::exception& exc) {std::cerr<<"Error: "<<exc.what()<<'\n';usage();return 1;}
}

// HEDGE paper's EXHAUST baseline: greedy using exact adaptive GBC marginal
// at every step. This is NOT exhaustive enumeration of all k-subsets and
// need not find the globally optimal group. Scores use ordered pairs.
// Inlined shared graph and sampling helpers for single-file Colab use.
// Shared unweighted graph, uniform shortest-path sampler, and group coverage.
// Every sampled hyperedge contains internal vertices only; unreachable pairs
// and direct edges yield an empty hyperedge. Vertex labels remain external.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace gbc {
using Label=std::int64_t;
using Path=std::vector<int>;
struct Graph {
    bool directed=false;
    std::vector<Label> labels;
    std::vector<std::vector<int>> out, incoming;
    std::size_t m=0;
    int n()const{return static_cast<int>(labels.size());}
};
inline Graph load(const std::string& filename,bool directed) {
    std::ifstream file(filename);
    if(!file)throw std::runtime_error("Cannot open graph: "+filename);
    std::vector<std::pair<Label,Label>> edges;
    std::vector<Label> labels;
    std::string line;
    while(std::getline(file,line)) {
        auto p=line.find_first_not_of(" \t\r\n");
        if(p==std::string::npos||line[p]=='#'||line[p]=='%')continue;
        std::istringstream row(line); Label u,v;
        if(!(row>>u>>v))throw std::runtime_error("Invalid edge: "+line);
        if(u==v)continue;
        if(!directed&&u>v)std::swap(u,v);
        edges.emplace_back(u,v);labels.push_back(u);labels.push_back(v);
    }
    std::sort(labels.begin(),labels.end());
    labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
    if(labels.size()<2)throw std::runtime_error("Graph requires at least two vertices");
    std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    Graph g;g.directed=directed;g.labels=std::move(labels);g.m=edges.size();
    g.out.resize(g.n());g.incoming.resize(g.n());
    for(const auto& [a,b]:edges) {
        int u=static_cast<int>(std::lower_bound(g.labels.begin(),g.labels.end(),a)-g.labels.begin());
        int v=static_cast<int>(std::lower_bound(g.labels.begin(),g.labels.end(),b)-g.labels.begin());
        g.out[u].push_back(v);g.incoming[v].push_back(u);
        if(!directed){g.out[v].push_back(u);g.incoming[u].push_back(v);}
    }
    for(auto& e:g.out)std::sort(e.begin(),e.end());
    for(auto& e:g.incoming)std::sort(e.begin(),e.end());
    return g;
}
struct Arguments {
    std::map<std::string,std::string> values;
    bool directed=false;
    Arguments(int argc,char** argv) {
        for(int i=1;i<argc;++i){
            std::string key=argv[i];
            if(key=="--directed"){directed=true;continue;}
            if(key=="--help"){
                std::cout<<"Required: --graph EDGELIST --k K; optional: --directed, --seed N, --output FILE. See README.md.\n";
                std::exit(0);
            }
            if(key.rfind("--",0)!=0||i+1==argc)throw std::invalid_argument("Invalid CLI option: "+key);
            values[key]=argv[++i];
        }
    }
    std::string get(const std::string& key,const std::string& fallback="")const {
        auto it=values.find(key);return it==values.end()?fallback:it->second;
    }
    int k(int n)const {
        int value=std::stoi(get("--k","0"));
        if(value<1||value>=n)throw std::invalid_argument("Expected 1 <= k < n");
        return value;
    }
};
inline std::string quoted(const std::string& s){
    std::ostringstream out;out<<'"';
    for(char ch:s){if(ch=='"'||ch=='\\')out<<'\\';if(ch=='\n'){out<<"\\n";continue;}out<<ch;}
    out<<'"';return out.str();
}
inline void emit(const Arguments& args,const Graph& g,const std::string& name,
                 const std::vector<int>& selected,const std::map<std::string,std::string>& extra){
    std::ostringstream out;out<<std::setprecision(18);
    out<<"{\n  \"method\": "<<quoted(name)<<",\n  \"graph\": "<<quoted(args.get("--graph"))
       <<",\n  \"directed\": "<<(g.directed?"true":"false")
       <<",\n  \"num_nodes\": "<<g.n()<<",\n  \"num_edges\": "<<g.m
       <<",\n  \"k\": "<<selected.size()<<",\n  \"nodes\": [";
    for(std::size_t i=0;i<selected.size();++i){if(i)out<<", ";out<<g.labels[selected[i]];}
    out<<"]";
    for(const auto& [key,value]:extra)out<<",\n  "<<quoted(key)<<": "<<value;
    out<<"\n}\n";
    std::cout<<out.str();
    if(!args.get("--output").empty()){
        std::ofstream file(args.get("--output"));
        if(!file)throw std::runtime_error("Cannot write output");
        file<<out.str();
    }
}
struct Workspace{
    std::vector<int> dist,order;
    std::vector<long double> sigma;
    explicit Workspace(int n):dist(n,-1),sigma(n,0){order.reserve(n);}
    void bfs(const Graph& g,int s,int stop_at=-1){
        std::fill(dist.begin(),dist.end(),-1);
        std::fill(sigma.begin(),sigma.end(),0.0L);
        order.clear();std::queue<int> q;q.push(s);dist[s]=0;sigma[s]=1;
        while(!q.empty()){
            int u=q.front();q.pop();
            if(stop_at>=0&&dist[stop_at]>=0&&dist[u]>=dist[stop_at])break;
            order.push_back(u);
            for(int v:g.out[u]){
                if(dist[v]<0){dist[v]=dist[u]+1;q.push(v);}
                if(dist[v]==dist[u]+1)sigma[v]+=sigma[u];
            }
        }
    }
};
inline Path sample(const Graph& g,Workspace& ws,std::mt19937_64& rng){
    std::uniform_int_distribution<int> node(0,g.n()-1);
    int s=node(rng),t=node(rng);while(t==s)t=node(rng);
    ws.bfs(g,s,t);Path path;
    if(ws.dist[t]<0)return path;
    while(t!=s){
        long double sum=0;
        for(int v:g.incoming[t])if(ws.dist[v]==ws.dist[t]-1)sum+=ws.sigma[v];
        if(sum<=0)throw std::logic_error("Missing predecessor");
        std::uniform_real_distribution<long double> pick(0,sum);
        long double x=pick(rng);int prev=-1;
        for(int v:g.incoming[t])if(ws.dist[v]==ws.dist[t]-1){
            prev=v;x-=ws.sigma[v];if(x<=0)break;
        }
        t=prev;
        if(t!=s)path.push_back(t);
    }
    return path;
}
inline std::vector<int> greedy(const std::vector<Path>& paths,int n,int k,std::size_t& covered){
    std::vector<std::vector<int>> incident(n);
    for(std::size_t i=0;i<paths.size();++i)for(int v:paths[i])incident[v].push_back(static_cast<int>(i));
    std::vector<int> gain(n),selected;std::vector<unsigned char> picked(n,0),hit(paths.size(),0);
    for(int v=0;v<n;++v)gain[v]=static_cast<int>(incident[v].size());
    covered=0;
    for(int step=0;step<k;++step){
        int best=-1;
        for(int v=0;v<n;++v)if(!picked[v]&&(best<0||gain[v]>gain[best]))best=v;
        picked[best]=1;selected.push_back(best);
        for(int id:incident[best])if(!hit[id]){
            hit[id]=1;++covered;
            for(int v:paths[id])--gain[v];
        }
    }
    return selected;
}
// Exact marginal B(C union {u})-B(C) for EVERY u, in one all-source pass.
// For one source, prefix[v] counts geodesics s->v avoiding internal C.
// suffix[v] sums reciprocal sigma_all[t] for suffixes from v to targets t.
// A group node may be a target but cannot be continued through.
inline std::vector<long double> exact_marginals(const Graph& g,const std::vector<unsigned char>& in_group){
    int n=g.n();std::vector<long double> result(n,0),prefix(n),suffix(n);
    Workspace ws(n);
    for(int s=0;s<n;++s){
        ws.bfs(g,s);
        std::fill(prefix.begin(),prefix.end(),0);prefix[s]=1;
        for(int v:ws.order){
            if(v!=s&&in_group[v])continue;
            for(int w:g.out[v])if(ws.dist[w]==ws.dist[v]+1)prefix[w]+=prefix[v];
        }
        std::fill(suffix.begin(),suffix.end(),0);
        for(auto it=ws.order.rbegin();it!=ws.order.rend();++it){
            int v=*it;
            for(int w:g.out[v])if(ws.dist[w]==ws.dist[v]+1)
                suffix[v]+=1.0L/ws.sigma[w]+(in_group[w]?0.0L:suffix[w]);
            if(v!=s&&!in_group[v])result[v]+=prefix[v]*suffix[v];
        }
    }
    return result;
}
inline int best_unselected(const std::vector<long double>& scores,const std::vector<unsigned char>& picked){
    int best=-1;
    for(std::size_t v=0;v<scores.size();++v)if(!picked[v]&&
        (best<0||scores[v]>scores[best]+1e-14L))best=static_cast<int>(v);
    return best;
}
} // namespace gbc

int main(int argc,char** argv){
    try{
        gbc::Arguments args(argc,argv);
        const auto g=gbc::load(args.get("--graph"),args.directed);
        const int k=args.k(g.n());
        std::vector<unsigned char> picked(g.n(),0);
        std::vector<int> selected;selected.reserve(k);
        long double score=0;
        const auto started=std::chrono::steady_clock::now();
        std::vector<long double> prefix_scores;
        std::vector<double> prefix_seconds;
        prefix_scores.reserve(k);prefix_seconds.reserve(k);
        for(int step=0;step<k;++step){
            auto gain=gbc::exact_marginals(g,picked);
            const int best=gbc::best_unselected(gain,picked);
            score+=gain[best];picked[best]=1;selected.push_back(best);
            prefix_scores.push_back(score);
            prefix_seconds.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now()-started).count());
            std::cerr<<"[EXHAUST] "<<step+1<<'/'<<k<<" selected "
                     <<g.labels[best]<<"; marginal="<<std::setprecision(14)
                     <<gain[best]<<"; raw="<<score<<'\n';
        }
        std::ostringstream raw,normalized;
        raw<<std::setprecision(18)<<score;
        normalized<<std::setprecision(18)<<score/(static_cast<long double>(g.n())*(g.n()-1));
        std::ostringstream score_array,time_array;
        score_array<<std::setprecision(18)<<'[';
        time_array<<std::setprecision(18)<<'[';
        for(int i=0;i<k;++i){
            if(i){score_array<<',';time_array<<',';}
            score_array<<prefix_scores[i];time_array<<prefix_seconds[i];
        }
        score_array<<']';time_array<<']';
        gbc::emit(args,g,"EXHAUST",selected,{{"raw_gbc",raw.str()},
                  {"normalized_gbc",normalized.str()},
                  {"prefix_raw_gbc",score_array.str()},
                  {"prefix_elapsed_seconds",time_array.str()},
                  {"status",gbc::quoted("exact_greedy")}});
        return 0;
    }catch(const std::exception& e){std::cerr<<"EXHAUST: "<<e.what()<<'\n';return 1;}
}

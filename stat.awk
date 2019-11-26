#graph;vertices;edges;source;kernel;time
#$1   ;$2      ;$3   ;$4    ;$5    ;$6

{if($1 != "graph"){
    result[$1][$5] += $6
    count[$1][$5]++
}}

END{
    for(graph in result){
        print graph
        for(kernel in result[graph]){
            result[graph][kernel] /= count[graph][kernel]
            print "\t" kernel " avg time: " result[graph][kernel]
        }
    }
}

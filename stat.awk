#graph;vertices;edges;source;kernel;time
#$1   ;$2      ;$3   ;$4    ;$5    ;$6

function max_of(array){
    max = 0
    for(x in array){
        if(array[x] > max){
            max = array[x]
        }
    }
    return max
}

BEGIN{
    FS=";"
    PREC=100
}

{if($1 != "graph"){
    graph = $1
    kernel = $5
    edges[graph] = $3 
    avg_degree[graph] = $3 / $2
    avg_time[graph][kernel] += $6
    avg_through[graph][kernel] += ($3 / $6) / 1000 #throughput in 10^9 edges per sec
    hmean[kernel] += (1 / avg_through[graph][kernel])
    count[graph][kernel]++
}}

END{
    for(graph in avg_time){
        print graph " avg. degree: " avg_degree[graph]
        for(kernel in avg_time[graph]){
            avg_time[graph][kernel] /= count[graph][kernel]
            avg_through[graph][kernel] /= count[graph][kernel]
            print "\t" kernel " run " count[graph][kernel] " times"
            print "\t\tavg time:\t" avg_time[graph][kernel] " (ms)"  
            print "\t\tavg throughput:\t" avg_through[graph][kernel] " (10^9 edges per sec)"
        }
    }
    print "NORMALIZED HMEAN"
    #delete hmean["Quadratic"]
    #delete hmean["Linear"]

    for(kernel in hmean){
        hmean[kernel] = 1 / hmean[kernel]
    }
    max = max_of(hmean)
    for(kernel in hmean){
        hmean[kernel] /= max
        print "\t" kernel "\t" hmean[kernel]
    }
}

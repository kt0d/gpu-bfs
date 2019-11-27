#graph;vertices;edges;source;kernel;time
#$1   ;$2      ;$3   ;$4    ;$5    ;$6

function max_of(array){
    max = -1
    for(x in array){
        if(array[x] > max){
            max = array[x]
        }
    }
    return max
}

BEGIN{
    FS=";"
    PREC=100 # i'm not sure if it even works
}

{if($1 != "graph"){ #ignore CSV headers
    graph = $1
    kernel = $5
    edges[graph] = $3 
    vertices[graph] = $2
    avg_degree[graph] = $3 / $2
    avg_time[graph][kernel] += $6
    avg_through[graph][kernel] += ($3 / $6) / 1000 #throughput in 10^6 edges per sec
    hmean[kernel] += (1 / avg_through[graph][kernel])
    count[graph][kernel]++
}}

END{
    for(graph in avg_time){
        printf "%-25s \nn=%-10'd m=%-12'd avg. degree=%s\n", graph, vertices[graph], edges[graph],  avg_degree[graph]
        for(kernel in avg_time[graph]){
            avg_time[graph][kernel] /= count[graph][kernel]
            avg_through[graph][kernel] /= count[graph][kernel]
            printf "\t%-22s %2s runs\n", kernel,  count[graph][kernel]
            print "\t\tavg time:\t" avg_time[graph][kernel] " (ms)"  
            print "\t\tavg throughput:\t" avg_through[graph][kernel] " (10^6 edges per sec)"
        }
    }
    print "\nNORMALIZED HMEAN"
    #delete hmean["Quadratic"]
    #delete hmean["Linear"]
    for(kernel in hmean){
        hmean[kernel] = 1 / hmean[kernel]
    }
    max = max_of(hmean)
    for(kernel in hmean){
        hmean[kernel] /= max
        printf "\t%-23s %s\n", kernel, hmean[kernel]
#	for(i = 0; i < hmean[kernel]*80; i+=3)
#		printf("░▒█")
#	printf("\n")

    }
}

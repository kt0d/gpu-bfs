gpu-bfs
=======
CUDA implementation of BFS graph traversal for GPU based on paper by Duane Merill, Michael Garland and Andrew Grimshaw.
https://research.nvidia.com/publication/high-performance-and-scalable-gpu-graph-traversal-0


Examples
--------
Run expand-contract, contract-expand, two-phase kernels and compare with simple CPU traversal
`./gpu-bfs -ECTc wikipedia-20070206.rb`

Getting help:
`./gpu-bfs --help`

Tested with:
------------
https://sparse.tamu.edu/RB/Gleich/wikipedia-20070206.tar.gz
https://sparse.tamu.edu/RB/DIMACS10/europe_osm.tar.gz
https://sparse.tamu.edu/RB/DIMACS10/coPapersCiteseer.tar.gz
https://sparse.tamu.edu/RB/vanHeukelum/cage15.tar.gz
https://sparse.tamu.edu/RB/Schenk/nlpkkt160.tar.gz
https://sparse.tamu.edu/RB/Zaoui/kkt_power.tar.gz
https://sparse.tamu.edu/RB/GHS_psdef/audikw_1.tar.gz
https://sparse.tamu.edu/RB/DIMACS10/hugebubbles-00020.tar.gz

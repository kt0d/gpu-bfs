projekt:
- [ ] dodać komentarze - wszędzie
 - [ ] .h na .hpp
 - [ ] ładne formatowanie - clang-format
 - [ ] profiler…
 - [ ] zmierzyć metryki jak Merill et al.
main:
 - [ ] argumenty - skorzystać z argp.h, być może
 - [ ] opcja wielokrotnego odpalenia z losowego wierzchołka
bfs:
 - [ ] initialize\_vertex\_queue - rozbić na wzór initialize\_edge\_queue
 - [ ] in\_queue\_count, out\_queue\_count - przerobić na memcpy albo chociaż pinned memory
kernele: 
 - [ ] zmniejszyć użycie rejestrów: możne je zbadać kompilując z opcją -dc i otwierając plik obiektowy przez 'cuobjdump -elf'
 - [ ] rozważyć zastąpienie int2 używanej do zwracania wyniku prefix scan własną strukturą z sensowną semantyką
 - [ ] shared memory - statycznie w funkcjach vs statycznie w kernelu vs dynamicznie
 - [ ] 2d bitmask (?)
contract-expand:
 - [ ] zrównoleglenie prefix sum
two-phe:
 - [ ] zrobić

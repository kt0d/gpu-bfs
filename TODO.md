do zapamiętania
- kompiacja z symbolami odpluskwiania (-G) powoduje dramatyczne zwolenienie
- kron jest MULTIGRAFEM
1. projekt:
   - [x] dodać komentarze - wszędzie
   - [ ] pozbyć się bezużytecznych komentarzy
   - [ ] .h na .hpp
   - [ ] ładne formatowanie - clang-format
   - [ ] profiler…
   - [ ] zmierzyć metryki jak Merill et al.
1. main:
   - [ ] rozbić na funkcje
   - [x] opcja wielokrotnego odpalenia z losowego wierzchołka
   - [ ] wiele grafów
1. csr\_matrix
   - [ ] dodać obsługę mutligrafów
   - [ ] w ogóle przerobić na csr\_graph czy coś
   - [ ] wczytywać .graph z DIMACS 10th challenge? - ale .rb są popularniejsze
1. bfs:
   - [ ] zwracać liczbę iteracji w bfs result
   - [x] initialize\_vertex\_queue - rozbić na wzór initialize\_edge\_queue
   - [x] in\_queue\_count, out\_queue\_count - przerobić na memcpy albo chociaż pinned memory; po przerobieniu na mapped pinned memory potężnie zwolniło, mogę jeszcze spróbwać memcpy ale na razie nie chcę; jest memcpy i jest ok
1. kernele: 
   - [ ] tile of input
   - [ ] fix block\_gather
   - [ ] by oszczędzić na rejestrach możnaw warp\_cull i status\_lookup zamiast bool zwracać albo prawidłowy vertex albo -1, pozbyć się remain
   - [ ] zmniejszyć użycie rejestrów: możne je zbadać kompilując z opcją -dc i otwierając plik obiektowy przez 'cuobjdump -elf'
   - [ ] rozważyć zastąpienie int2 używanej do zwracania wyniku prefix scan własną strukturą z sensowną semantyką
   - [ ] shared memory - statycznie w funkcjach vs statycznie w kernelu vs dynamicznie
   - [ ] 2d bitmask (?)
1. contract-expand:
   - [ ] zrównoleglenie prefix sum
1. two-phe:
   - [ ] zrobić

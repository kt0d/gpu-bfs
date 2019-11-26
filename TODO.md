IMPORTANT
w contract expand jest bug objawiający się dla grafu kron_g500-logn20 ze źródla 668976. wydaje się że przyczyną może być źle działająca unified memory, a konkretnie w run_contract_expand out_queue_count chociaż zerowanie nie przenosi się do GPU
ZATEM muszę użyć cudamemcpy jak człowiek

1. projekt:
   - [ ] dodać komentarze - wszędzie
   - [ ] .h na .hpp
   - [ ] ładne formatowanie - clang-format
   - [ ] profiler…
   - [ ] zmierzyć metryki jak Merill et al.
1. main:
   - [ ] argumenty - skorzystać z argp.h, być może; chyba jednak tego nie zrobię
   - [x] opcja wielokrotnego odpalenia z losowego wierzchołka
   - [ ] wiele grafów
1. bfs:
   - [x] initialize\_vertex\_queue - rozbić na wzór initialize\_edge\_queue
   - [x] in\_queue\_count, out\_queue\_count - przerobić na memcpy albo chociaż pinned memory; po przerobieniu na mapped pinned memory potężnie zwolniło, mogę jeszcze spróbwać memcpy ale na razie nie chcę
1. kernele: 
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

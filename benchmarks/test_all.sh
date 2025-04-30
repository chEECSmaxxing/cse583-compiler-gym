
# strings=("apple" "banana" "cherry" "date" )
# "bitcount" "qsort" "susan"
strings=("bzip2" "jpeg-c" "lame" "tiff2bw" "tiff2rgba" "tiffdither" "tiffmedian" "dijkstra" "patricia" "stringsearch" "blowfish" "rijndael" "sha" "adpcm" "crc32" "gsm")

for item in "${strings[@]}"; do
    echo cbench/$item SARSA
    python sarsa_rl.py --benchmark=cbench-v1/$item --episodes=1000 | tee results/sarsa_$item.txt
    echo cbench/$item TABULARQ
    python tabular_q.py --benchmark=cbench-v1/$item --episodes=1000 | tee results/tabular_$item.txt
done
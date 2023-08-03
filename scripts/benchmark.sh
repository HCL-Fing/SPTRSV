
echo "double" >> resultados.txt
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/chipcool0.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/hollywood-2009.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/nlpkkt160.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/road_central.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/road_usa.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/ship_003.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/webbase-1M.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/wiki-Talk.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/crankseg_1.mtx
./sptrsv_double /home/gpgpu/users/edufrechou/matrices/sptrsv/cant.mtx

echo "simple" >> resultados.txt
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/chipcool0.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/hollywood-2009.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/nlpkkt160.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/road_central.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/road_usa.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/ship_003.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/webbase-1M.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/wiki-Talk.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/crankseg_1.mtx
./sptrsv_float /home/gpgpu/users/edufrechou/matrices/sptrsv/cant.mtx
#!/bing/bash
matrix=(chipcool0 hollywood-2009 nlpkkt160 road_central road_usa ship_003 webbase-1M wiki-Talk crankseg_1 cant cvxbqp1 ncvxqp3 luxembourg_osm rajat29 bayer01 circuit5M_dc exdata_1 CO 2D_54019_highK cond-mat TSOPF_RS_b300_c2 LeGresley_87936)




#matrix=(kmer_U1a europe_osm hugebubbles-00020 kmer_A2a kmer_V1r kmer_P1a kmer_V2a road_usa nlpkkt240 uk-2005 it-2004 arabic-2005 mawi_201512020000 mawi_201512020030 mawi_201512020130 mawi_201512020330 twitter7 sk-2005)

matrix1=(rajat29)
folder="log"
mode=$1
user=$2
pass=$3
echo "Running in mode: $mode"
export OMP_PROC_BIND
case $mode in
1)	#Grupo de prueba Analisis y solvers
	rm sptrsv_double_main

	echo "Tiempo solver multirow con los dos análisis" 

	file="Tiempos-3090-grupo-chico.csv"
        make OLD=0 LOG_FILE="\\\"$folder/$file\\\"" TIMER_SOLVERS=1

    #make LOG_FILE="\\\"$folder/$file\\\""
	rm "$folder/$file"
	echo "Matriz,Analisis1,Analisis2,Solver-MR,Solver-Lev,Solver-Format" >> "$folder/$file"
	for i in "${matrix[@]}"
	do    

	 #   echo -n "$i" >> "t_mem.csv"
	   
	echo -n "$i" >> "$folder/$file"
	timeout 40 ./sptrsv_double_main /media/matrices/ssget/MM/todas/"$i".mtx #"$folder/$file"
	echo "" >> "$folder/$file"   
	#    echo "" >> "t_mem.csv"	
	done
;;
2)
	#Medir tiempos en 3090
	
	file="Tiempos-3090.csv"
	rm sptrsv_double_main
	make OLD=0 LOG_FILE="\\\"$folder/$file\\\"" TIMER_SOLVERS=1 # EN MAKEFILE PRINT_TIME_ANALYSIS=1
	rm "$folder/$file"
	
	#rm "fallos.txt"
		
	echo "Matriz,Analisis1,Analisis2,Solver-MR,Solver-Lev,Solver-Format" >> "$folder/$file"

	matrices=$(cat lista-ejecucion.txt)


    for m in $matrices
	do
		nombre_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\2/');
		echo -n -e "$m" >>"$folder/$file" 
	   	timeout 1200 ./sptrsv_double_main /media/matrices/ssget/MM/todas/"$m" || ( echo "$m" >> fallos.txt) 
		echo " " >> "$folder/$file"
	done
;;
3)
# Saca nombre de las matrices cuadradas con dim > 10k 
	search_dir=/media/matrices/ssget/MM/todas/


	make MAIN=matricesMayores
	rm lista2.txt
	#rm datos_matrices.csv
	for i in "$search_dir"/*
	do
		if timeout 40 ./sptrsv_double_matricesMayores $i;
		then 
 	               	name=${i##*/}
#        	        echo -n "$name" >> datos_matrices.csv
			echo "$name" >> lista2.txt 
		fi

	done
	
;;
4)
# Saca métricas de las matrices cuadradas con dim > 10k a partir de lista
search_dir=/media/matrices/ssget/MM/todas/



        matrices=$(cat lista-ejecucion.txt)

        #make MAIN=matricesMayores
        #rm lista.txt
        #rm datos_matrices.csv
        for i in $matrices
        do
                if timeout 400 ./sptrsv_double_matricesMayores "$search_dir/$i";
                then
                        name=${i##*/}
                        echo "$name" >> datos_matrices.csv
                        #echo "$name" >> lista.txt
                fi

        done

;;
11)
	
	file="tiempos_todas_matrices_solvers_3090.csv"
	make TIMER_SOLVERS=1 LOG_FILE="\\\"$folder/$file\\\"" 
	echo "Matriz,DIM,Levels,NNZ_L,Analisis_Mult, Analisis_Ord,Solver Simple,Solver Order,Solver Multirow,Analisis_Cusp,Solver Cusp" >> "$folder/$file"

	matrices=$(cat listaMayores.txt)
	matdir=dirMatrices
	#rm -r dirMatrices
	#mkdir dirMatrices

	for m in $matrices
	do
	 	#grupo_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\1/');
	 	nombre_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\2/');

	 	cd ~/sptrsv

	 	sshpass -p "$pass" scp "$user"@labgpu03:/media/matrices/ssget/MM/cuadradas/$nombre_mat dirMatrices
	 	echo -n -e "\n $nombre_mat" >> "$folder/$file" 
		timeout 40 ./sptrsv_double dirMatrices/$nombre_mat || echo "$nombre_mat" >> fallos.txt

		cd $matdir
		rm $nombre_mat
	done

	#rm -r dirMatrices
;;
12)
	#Ejemplo de ejecución en lab2 (con grupo chico)
	file="kernel_viejo_lab2.csv"
	rm $file
	make LOG_FILE="\\\"$folder/$file\\\""
	echo "Matriz,Init,Kernel,Dfr_info,Max,Iorder/Size calculation,Warp Info,Copy to devicce,Free" >> "$folder/$file"

	matdir=dirMatrices
	#rm -r dirMatrices
	#mkdir dirMatrices

	for m in "${matrix[@]}"
	do
	 	#nombre_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\2/');

	 	cd ~/sptrsv

	 	sshpass -p "$pass" scp "$user"@labgpu03:/media/matrices/ssget/MM/todas/$m.mtx dirMatrices
	 	echo -n -e "$m.mtx" >> "$folder/$file" 
		timeout 40 ./sptrsv_double dirMatrices/$m.mtx || echo "$m.mtx" >> fallos.txt

		cd $matdir
		rm $m.mtx
	done
esac


#scl enable devtoolset-8 bash
#compute-sanitizer ./sptrsv_double /media/matrices/ssget/MM/todas/ibm_matrix_2.mtx
#export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1

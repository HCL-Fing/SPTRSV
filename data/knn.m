data = csvread('data/datos1.csv');
tiempos =  csvread('data/tiempos.csv');

Xt = data(1:floor(.8*length(data))               ,1:size(data,2)-1);  
Yt = data(1:floor(.8*length(data))               ,  size(data,2)  );
Xp = data(  floor(.8*length(data))+1:length(data),1:size(data,2)-1);
Yp = data(  floor(.8*length(data))+1:length(data),  size(data,2)  );
Tp = tiempos(floor(.8*length(tiempos))+1:length(tiempos),:);

% Xt = data(:,1:size(data,2)-1);  
% Yt = data(:,size(data,2)); 

k = floor(sqrt(length(Yt)));

if mod(k,2)==0
	k = k+1;
end

k

% mdlk = fitcknn(Xt,Yt,'NumNeighbors',3,'Standardize',1);

mdlk = fitcknn(Xt,Yt,'NumNeighbors',k,'NSMethod','exhaustive','Distance','minkowski','Standardize',1);

cvmdl = crossval(mdlk);

loss = kfoldLoss(cvmdl)

label = predict(mdlk, Xp);

aciertos = (length(Yp) - nnz(label - Yp))/length(Yp)

T_error = zeros(length(Yp),1);

for i = 1 : length(Yp)
	T_error(i) =( Tp(i,label(i)) - Tp(i,Yp(i)) ) / Tp(i,Yp(i)) ;
end

ARE = sum(T_error)/length(T_error)
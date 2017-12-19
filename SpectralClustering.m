%Implementation of the spectral clustering k-eigenvector algorithm
%described in "On Spectral Clustering: Analysis and an algorithm" 
%by Ng et al. 2001

S = csvread('Path-to-csv-file');
k = 9;      %Number of clusters
sig = 1;    %Sigma

for i=1:size(S,1)    
    for j=1:size(S,1)
        avst = sqrt((S(i,1) - S(j,1))^2 + (S(i,2) - S(j,2))^2);     %Creating the affinity matrix A
        A(i,j) = exp(-avst/(2*sig^2));
    end
end

D = zeros(size(A));
for i=1:size(A,1)       %Step 2, creating the diagonal matrix D
    D(i,i) = sum(A(i,:));
end

L = D^(-1/2) * A * D^(-1/2);
[Vektorer,Varden] = eig((L * L')/2);      %Eigenvalues and Eigenvectors of L
stora = Vektorer(:,(size(Vektorer,1)-(k-1)): size(Vektorer,1));      %Stacking the k largest eigenvectors in columns

Y=stora./repmat(sqrt(sum(stora.^2,2)),[1 k]);

IDX = kmeans(Y, k);     %K-means clustering

%Plotting the results
figure,
hold on;
for i=1:size(IDX,1)
    if IDX(i,1) == 1
        plot(S(i,1),S(i,2),'r+');
     elseif IDX(i,1) == 2
         plot(S(i,1),S(i,2),'y+');
    elseif IDX(i,1) == 3
        plot(S(i,1),S(i,2),'g+');
    elseif IDX(i,1) == 4
        plot(S(i,1),S(i,2),'c+'); 
    elseif IDX(i,1) == 5
        plot(S(i,1),S(i,2),'b+'); 
    elseif IDX(i,1) == 6
        plot(S(i,1),S(i,2),'m+'); 
    elseif IDX(i,1) == 7
        plot(S(i,1),S(i,2),'k+'); 
    elseif IDX(i,1) == 8
        plot(S(i,1),S(i,2),'r+'); 
    elseif IDX(i,1) == 9
        plot(S(i,1),S(i,2),'y+'); 
    end
end
hold off;
title('Resulting clusters');
grid on;shg






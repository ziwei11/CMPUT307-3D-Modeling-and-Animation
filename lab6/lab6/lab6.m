fig1=figure('Name','3d_sphere.mat','NumberTitle','off');
load('3d_sphere.mat');
[three_d_sphere_cov_matrix,three_d_sphere_eigenvalues,three_d_sphere_eigenvectors] = PCA(X);
three_d_sphere_eigenvalues
three_d_sphere_eigenvectors
plot3d_pca(X);

fig2=figure('Name','teapot.mat','NumberTitle','off');
load('teapot.mat');
[teapot_cov_matrix,teapot_eigenvalues,teapot_eigenvectors] = PCA(X);
teapot_eigenvalues
teapot_eigenvectors
plot3d_pca(X);

fig3=figure('Name','bun_zipper.mat','NumberTitle','off');
load('bun_zipper.mat');
[bun_zipper_cov_matrix,bun_zipper_eigenvalues,bun_zipper_eigenvectors] = PCA(X);
bun_zipper_eigenvalues
bun_zipper_eigenvectors
plot3d_pca(X);

fig4=figure('Name','eclipse1.mat','NumberTitle','off');
load('eclipse1.mat');
[eclipse1_cov_matrix,eclipse1_eigenvalues,eclipse1_eigenvectors] = PCA(X);
eclipse1_eigenvalues
eclipse1_eigenvectors
plot2d_pca(X);

fig5=figure('Name','eclipse2.mat','NumberTitle','off');
load('eclipse2.mat');
[eclipse2_cov_matrix,eclipse2_eigenvalues,eclipse2_eigenvectors] = PCA(X);
eclipse2_eigenvalues
eclipse2_eigenvectors
plot2d_pca(X);

fig6=figure('Name','eclipse3.mat','NumberTitle','off');
load('eclipse3.mat');
[eclipse3_cov_matrix,eclipse3_eigenvalues,eclipse3_eigenvectors] = PCA(X);
eclipse3_eigenvalues
eclipse3_eigenvectors
plot2d_pca(X);


function [covariance_matrix,eigenvalues,eigenvectors] = PCA(X)
% use function center(X) to get X_centered
[X_centered,~] = center(X);
% get covariance_matrix by cov()
covariance_matrix = cov(X_centered);
% get eigenvectors and eigenvalues by eig()
[eigenvectors,eigenvalues] = eig(covariance_matrix);
end


function [X_centered,centroid] = center(X)
% get centroid by mean()
centroid = mean(X);
% X_centered = X-centroid
X_centered = X-centroid;
end


function [] = plot2d_pca(X)
x = X(1:end,1);
y = X(1:end,2);
% plot original dataset
plot(x,y,'.');
hold on;
% get centroid
[~,centroid] = center(X);
% get eigenvalues and eigenvectors
[~,~,eigenvectors] = PCA(X);

% scale the principal component with its variance
% plot the principal components
plot([centroid(1),centroid(1)+eigenvectors(1,2)],[centroid(2),centroid(2)+eigenvectors(2,2)],'B');
hold on;
plot([centroid(1),centroid(1)+eigenvectors(1,1)],[centroid(2),centroid(2)+eigenvectors(2,1)],'R');
end


function [] = plot3d_pca(X)
x = X(1:end,1);
y = X(1:end,2);
z = X(1:end,3);
% plot original dataset
plot3(x,y,z,'.');
hold on;
% get centroid
[~,centroid] = center(X);
% get eigenvalues and eigenvectors
[~,~,eigenvectors] = PCA(X);

% scale the principal component with its variance
% plot the principal components
plot3([centroid(1),centroid(1)+eigenvectors(1,3)],[centroid(2),centroid(2)+eigenvectors(2,3)],[centroid(3),centroid(3)+eigenvectors(3,3)],'B');
hold on;
plot3([centroid(1),centroid(1)+eigenvectors(1,2)],[centroid(2),centroid(2)+eigenvectors(2,2)],[centroid(3),centroid(3)+eigenvectors(3,2)],'R');
hold on;
plot3([centroid(1),centroid(1)+eigenvectors(1,1)],[centroid(2),centroid(2)+eigenvectors(2,1)],[centroid(3),centroid(3)+eigenvectors(3,1)],'G');
end
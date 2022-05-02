% we will perform a rotation followed by a translation, and then in reverse 
% order and compare the results, both using the homogeneous system
d_x = 1; % howmuchever you want to translate in horizontal direction
d_y = 1; % howmuchever you want to translate in vertical direction

% create original box
% first row is horizontal and second row is vertical coordinates
my_pts = [3 3 4 4 3;3 4 4 3 3];

% write code to plot the box
fig1=figure(1);
plot(my_pts(1,1:end),my_pts(2,1:end),'b*-');

% write code here to create your 2D rotation matrix my_rot 
a = -1/4*pi;
my_rot = [cos(a),-sin(a);sin(a),cos(a)];
[m,n] = size(my_rot);

% write code to create your Homogeneous 2D Translation matrix hom_trans using d_x & d_y 
trans = zeros(2,1);
trans(1,1) = d_x;
trans(2,1) = d_y;
hom_trans = eye(3,3);
hom_trans(1:2,3) = trans;

% Perform Compound transformation
% write code to construct your 2D Homogeneous Rotation Matrix using my_rot and store the result in hom_rot
% HINT: start with a 3x3 identity matrix and replace a part of it with my_rot to create hom_rot 
hom_rot = eye(m+1,n+1);
hom_rot(1:m,1:m) = my_rot;

% write code to convert my_pts to the homogeneous system and store the result in hom_my_pts
[m,n] = size(my_pts);
row = ones(1, n);
hom_my_pts = [my_pts;row];

% write code to perform in a single compound transformation: translation (hom_trans) followed by rotation (hom_rot) on hom_my_pts, and store the result in trans_my_pts 
trans_my_pts = (hom_trans*hom_rot)*hom_my_pts;

% write code to plot the transformed box (output) which has to be done in Cartesian, so... 
% cut out the X, Y points and ignore the 3rd dimension
hold on
plot(trans_my_pts(1,1:end),trans_my_pts(2,1:end),'r*-');
axis([2 8 -2 5]); % just to make the plot nicely visible

% Now, let us reverse the order of rotation and translation and compare 
fig2=figure(2);
plot(my_pts(1,1:end),my_pts(2,1:end),'b*-');

% write code to perform in a single compound transformation: rotation followed by translation, and store the result in trans_my_pts
trans_my_pts = (hom_rot*hom_trans)*hom_my_pts;
hold on

plot(trans_my_pts(1,1:end),trans_my_pts(2,1:end),'r*-');
axis([2 8 -2 5]); % just to make the plot nicely visible
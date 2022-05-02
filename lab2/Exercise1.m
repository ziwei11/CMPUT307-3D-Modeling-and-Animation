% create box
% first row is horizontal coordinates; second row is vertical coordinates
my_pts = [2 2 3 3 2;2 3 3 2 2];

% write code here to display the original box
fig1=figure(1);
plot(my_pts(1,1:end),my_pts(2,1:end),'b*-');

% write code here to create your 2D rotation matrix my_rot
a = -1/6*pi;
my_rot = [cos(a),-sin(a);sin(a),cos(a)];

% write code to perform rotation using my_rot and my_pts and store the result in my_rot_pts 
my_rot_pts = my_rot*my_pts;
hold on;

% write code to plot output
plot(my_rot_pts(1,1:end),my_rot_pts(2,1:end),'r*-');
axis([1.5 4.5 0 3.5]);
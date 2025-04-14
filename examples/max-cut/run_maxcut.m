% Author: Richard Y Zhang <ryz@illinois.edu>
% Date:   April 24th, 2020
% This program is licenced under the BSD 2-Clause licence,
% contained in the LICENCE file in this directory.

load Ybus_39
% load Ybus_118

% Set up MAXCUT problem
% Run experiment
[c,A,lb,ub] = genMAXCUT(Ybus,3);
[A2,b2,c2,K2,info] = ctc(c,A,lb,ub);

disp('Problem size:');
disp([size(A2,1), size(A2,2)]);
disp('Size of b:');
disp(size(b2));
disp('Size of c:');
disp(size(c2));
disp('Content of K:');
disp(K2);
disp('Content of info:');
disp(info);
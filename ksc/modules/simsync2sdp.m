function SDP = simsync2sdp(C,lam)
if nargin < 2
    fprintf('No scale regularization.\n')
    reg = false;
else
    fprintf('Add scale regularization with factor %3.2f.\n',lam);
    reg = true;
end
fprintf('generate sdpt3 and sedumi data ...')
tic;
%% return standard sdpt3 data and sedumi data of the SIM-Sync SDP
n = size(C,1)/3;

blk = {'s',3*n};

Acell = {};
b = [];

% the leading 3x3 block equals to identity
for i = 1:3
    for j = i:3
        A = sparse([i],[j],[1],3*n,3*n);
        A = (A + A')/2; % make it symmetric
        Acell = [Acell;{A}];
        if i == j
            b = [b;1];
        else
            b = [b;0];
        end
    end
end

% all the other blocks are scaled identity
for k = 1:n-1
    % diagonal equal
    A1 = sparse([3*k+1, 3*k+2],[3*k+1, 3*k+2],[1,-1],3*n,3*n);
    A2 = sparse([3*k+1, 3*k+3],[3*k+1, 3*k+3],[1,-1],3*n,3*n);
    % off-diagonal zero
    A3 = sparse([3*k+1, 3*k+2],[3*k+2, 3*k+1],[1/2,1/2],3*n,3*n);
    A4 = sparse([3*k+1, 3*k+3],[3*k+3, 3*k+1],[1/2,1/2],3*n,3*n);
    A5 = sparse([3*k+2, 3*k+3],[3*k+3, 3*k+2],[1/2,1/2],3*n,3*n);
    Acell = [Acell;{A1;A2;A3;A4;A5}];
    b = [b;zeros(5,1)];
end

b = sparse(b);

At = sparsesvec(blk,Acell);
At_sedumi = sparsevec(blk,Acell);

sdpt3.blk = blk;
sdpt3.At  = {At};
sdpt3.b   = b;
sdpt3.C   = {C};

K.s = 3*n;
sedumi.K = K;
sedumi.At = At_sedumi;
sedumi.b  = b;
sedumi.c  = sparsevec(blk,{C});

SDP.sdpt3 = sdpt3;
SDP.sedumi = sedumi;

%% add scale regularization
if reg
    %% modify sdpt3 format
    blk = sdpt3.blk;
    At  = sdpt3.At;
    b   = sdpt3.b;
    C   = sdpt3.C;
    % add cones
    for i = 1:n-1
        blk = [blk;{'s',2}];
    end
    blk = [blk; {'l',n-1}];
    
    % modify the objective function to add lam * sum(z)
    for i = 1:n-1
        C = [C;{zeros(2,2)}];
    end
    C = [C;{lam*ones(n-1,1)}];
    % add linear constraints
    for i = 1:n-1
        At = [At; {sparse(triangle_number(2),length(b))}];
    end
    At = [At;{sparse(n-1,length(b))}];

    At_sedumi = sedumi.At;
    At_sedumi = [sparse(n-1,size(At_sedumi,2));...
                 At_sedumi; ...
                 sparse(4*(n-1),size(At_sedumi,2))];
    
    for i = 1:n-1
        A0cell = {};
        Ascell = {};

        A0cell = [A0cell; { sparse(1,1,1,3*n,3*n) }];
        Ascell = [Ascell; { sparse(1,1,-1,2,2) }];
        A0cell = [A0cell; { sparse([3*i+1:3*i+3,1],[3*i+1:3*i+3,1],[1/3*ones(1,3),-1],3*n,3*n) }];
        Ascell = [Ascell; { sparse([1,2],[2,1],[-1/2,-1/2],2,2)}];
        Ascell = [Ascell; { sparse(2,2,-1,2,2)}];
        
        At{1,1} = [At{1,1}, sparsesvec(blk(1,:), A0cell), sparse(triangle_number(3*n),1)];
        At{1+i,1} = [At{1+i,1}, sparsesvec(blk(1+i,:), Ascell)];
        At{end,1} = [At{end,1}, sparse(n-1,2), sparse(i,1,1,n-1,1)];
        
        % sdpt3
        for j = 1:n-1
            if j ~= i
                At{1+j,1} = [At{1+j,1}, sparse(triangle_number(2),3)];
            end
        end
        % sedumi
        At0 = [sparsevec(blk(1,:), A0cell), sparse((3*n)^2,1)];
        Ats = [];
        for j = 1:n-1
            if j == i
                Ats = [Ats; sparsevec(blk(1+i,:), Ascell)];
            else
                Ats = [Ats; sparse(4,3)];
            end
        end
        Atl = [sparse(n-1,2),sparse(i,1,1,n-1,1)];

        At_sedumi = [At_sedumi, [Atl; At0; Ats]];

    end

    b = [b;zeros(3*(n-1),1)];
    sdpt3.At = At;
    sdpt3.b  = b;
    sdpt3.C  = C;
    sdpt3.blk = blk;
    SDP.sdpt3 = sdpt3;

    %% modify sedumi format
    sedumi.b = b;
    sedumi.K.l = n-1;
    sedumi.K.s = [sedumi.K.s,2*ones(1,n-1)];
    sedumi.c = [lam*ones(n-1,1); sedumi.c; sparse(4*(n-1),1)];
    sedumi.At = At_sedumi;
    SDP.sedumi = sedumi;
end

toc;
end


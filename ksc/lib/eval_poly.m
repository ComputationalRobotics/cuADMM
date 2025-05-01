function v = eval_poly(p,x,xval)
assert(length(x)==size(xval,1),"check dimension");
v = zeros(size(xval,2),1);
for i = 1:size(xval,2)
    v(i) = double(subs(p,x,xval(:,i)));
end
end

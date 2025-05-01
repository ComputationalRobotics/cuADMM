function monomial = mymonomial(vars, degree)
	deg = zeros(degree, 1);
	monomial = msspoly(1);
    if (degree == 0)
        return
    end
	n = length(vars);
	while true
		for i = 1 : degree
			if (deg(i) ~= n)
				v = deg(i) + 1;
				for j = 1 : i
					deg(j) = v;
				end
				break
			end
		end
		m = msspoly(1);
		for i = 1 : degree
			if (deg(i) ~= 0)
				m = m * vars(deg(i));
			end
		end
		monomial = [ monomial; m ];
        if (deg(degree) == n)
            return
	end
end

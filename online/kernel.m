function k=kernel(xi, xj)
    for i=1:size(xi,1)
        k(i,1)=xi(i,:)*xj';
    end
end
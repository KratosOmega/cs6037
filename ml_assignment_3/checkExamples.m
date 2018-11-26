function [add, a, b, Error, w] = checkExamples(i2, Y, a, tol, C, Error, eps, K, b, w, X_train)

% Init
add=0;
y2 = Y(i2);
alph2 = a(i2); % lagrande multiplier for i2

% SVM output on point[i2] - y2
if ((alph2 < tol) || (alph2 > C - tol))
    point_i2 = sum(Y' .* a .* K(i2, :)) - b;
    E2 = point_i2 - y2;
else
    E2 = Error(i2);
end

r2 = E2 * y2;

if (((r2 < -tol) && (alph2 < C)) || ((r2 > tol) && (alph2 > 0)))
    
    indx = find((Error > tol) & (Error < C - tol));
    
    if(~isempty(indx)) && (E2>0)
        [osef, i1] = max(Error);
        [check, a, Error, b, w] = takeStep(i1, i2, a, Y, b, eps, Error, C, K, E2, tol, w, X_train);
        
        if check
            add=1;
            return;
        end
    elseif (~isempty(indx)) && (E2<0)
        [osef, i1] = min(Error);
        [check, a, Error, b, w] = takeStep(i1, i2, a, Y, b, eps, Error, C, K, E2, tol, w, X_train);
        
        if check
            add=1;
            return;
        end
    end
    
    % Loop over all non-zero and non-C alpha, starting at randmom points
    if (~isempty(indx))
        
        startPoint = randi(length(indx));
        
        % Reorder the indx matrix
        indx = [indx(startPoint:end) indx(1:startPoint-1)];
        
        for i1=indx
            [check, a, Error, b, w] = takeStep(i1, i2, a, Y, b, eps, Error, C, K, E2, tol, w, X_train);
            
            if check
                add = 1; 
                return
            end
        end
    end
    % Loop over all lagrange multiplier elements, starting at randmom points
    for i1=1:length(a)
        [check, a, Error, b, w]=takeStep(i1, i2, a, Y, b, eps, Error, C, K, E2, tol, w, X_train);
        
        if check
            add = 1;
            return
        end
    end
end



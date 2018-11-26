function [out, a, Error, b, w] = takeStep(i1, i2, a, Y, b, eps, Error, C, K, E2, tol, w, X_train)

    out = 0;

    if(i1 == i2)
        return;
    end
    
    % Lagrange multiplier for i1
    alph1 = a(i1);
    alph2 = a(i2);
    
    % Check if alpha1 is on-bound-->[YES; NO]=[Evaluate E1 from SVM function; Take E1 from cache Error vector]
    if ((alph1 < tol) || (alph1 > C - tol))
        
        point_i1= sum(Y' .* a .* K(i1, :)) - b;
        E1 = point_i1 - Y(i1);
    else
        E1 = Error(i1);
    end
    
    % Computing the linear constraint bound, L, H (See literature for details)
    s = Y(i1) * Y(i2);
    
    if (s > 0)
        L = max(0, alph2 + alph1 - C);
        H = min(C,alph1 + alph2);
    else
        L = max(0, alph2 - alph1);
        H = min(C, C + alph2 - alph1);
    end
    
    if (L == H)
        return;
    end
    
    k11 = K(i1, i1);
    k12 = K(i1, i2);
    k22 = K(i2, i2);
    
    eta = 2 * k12 - k11 - k22;
    gamma = a(i1) + s * a(i2);
    
    if (eta < 0)
        a2 = alph2 - Y(i2) * (E1 - E2) / eta;
        if (a2 < L)
            a2 = L;
        elseif (a2 > H)
            a2 = H;
        end
    else
        Lobj = -s*L+L-0.5*k11*(gamma-s*L)^2-.5*k22*L^2-s*k12*(gamma-s*L)*L-Y(i1)...
            *(gamma-s*L)*(sum(Y' .* a .* K(i1, :)) - b+b-Y(i1)*alph1*k11-Y(i2)*alph2*K(i2,i1))...
            -Y(2)*L*(sum(Y' .* a .* K(i2, :)) - b+b-Y(i1)*alph1*k12-Y(i2)*alph2*k22);

        
        Hobj = -s*H+H-0.5*k11*(gamma-s*H)^2-.5*k22*H^2-s*k12*(gamma-s*H)*H-Y(i1)...
            *(gamma-s*H)*(sum(Y' .* a .* K(i1, :)) - b+b-Y(i1)*alph1*k11-Y(i2)*alph2*K(i2,i1))...
            -Y(2)*H*(sum(Y' .* a .* K(i1, :)) - b+b-Y(i1)*alph1*k12-Y(i2)*alph2*K(i2,i1));
        
        if (Lobj < Hobj - eps)
            a2 = L;
        elseif (Lobj > Hobj + eps)
            a2 = H;
        else
            a2 = alph2;
        end
    end
    
    if(a2 < 1e-8)
        a2 = 0;
    elseif (a2 > C-1e-8)
        a2 = C;
    end
    
    %%if the change in the first lagrange multipler is small return zero
    if (abs(a2 - alph2) < eps * (a2 + alph2 + eps))
        return
    end
    a1 = alph1 + s * (alph2 -a2);
    
   %%evaluate threshold or b AND updating the lagrange multipliers and
   %%caches errors
   b1=E1+Y(i1)*(a1-alph1)*k11+Y(i2)*(a2-alph2)*k12+b;
   b2=E2+Y(i1)*(a1-alph1)*k12+Y(i2)*(a2-alph2)*k22+b;
   bo=b;
   b=(b1+b2)/2;
   
   Error=Error+Y(i1)*(a1-alph1).*K(i1,:)+Y(i2)*(a2-alph2).*K(i2,:)+bo-b;
   Error(i1)=0;Error(i2)=0;
   a(i1)=a1;
   a(i2)=a2;
   
   w
   w = w + Y(i1) * (a1-alph1) .* X_train(i1,:) + Y(i2) * (a2-alph2) .* X_train(i2,:);
   
   out=1;
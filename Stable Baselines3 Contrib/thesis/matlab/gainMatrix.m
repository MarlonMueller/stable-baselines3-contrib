function K = gainMatrix()

    %Mass
    m = 1.0;
    %Length
    l = 1.0;
    %Gravity
    g = 9.81;
    %Timestep
    ts = 0.05;
        
    % Control matrix
    A = [0 1; g/l, 0];
    % System matrix
    B = [0; 1/(m*l^2)];
    % State weights
    Q = eye(2);
    % Control weights
    R = 1;
    % LQR
    [K, ~, ~] = lqr(A,B,Q,R);
    
end
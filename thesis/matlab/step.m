function x = step(x, u, w)

    % TimeStep
    ts = 0.05;

    % Maximal torque restriction
    % max_torque = 30.898877999566082;
    
    % Maximal angular acceleration restriction
    % Note: Not clipped within ode45
    %max_thdot = 10.0;

    % Clip torgue
    % u = min(max(u, -max_torque), max_torque);
            
    % Clip angular acceleration
    % x(2) = min(max(x(2), -max_thdot), max_thdot);

     %See also: https://www.mathworks.com/help/matlab/ref/ode45.html
    [~, x] = ode45(@(t, x) mathematicalPendulum(x, u, w), [0 ts], x);
    x = x(end, :);
    
end
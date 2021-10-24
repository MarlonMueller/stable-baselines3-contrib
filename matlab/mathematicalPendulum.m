function f = mathematicalPendulum(x, u, w)

    %Mass
    m = 1.0;
    %Length
    l = 1.0;
    %Gravity
    g = 9.81;

    %State space representation
    f(1,1) = x(2) + w(1);
    f(2,1) = g/l * sin(x(1)) + (1.0/(m*l^2)) * u(1) + w(2);

end
function [b, v] = regionOfAttraction()

    % Maximal torque restriction
    max_torque = 30.898877999566082;

    % Maximal angular acceleration searched
    max_thdot = 5*pi;

    % Define benchmark
    f = 'mathematicalPendulum';

    % Define algorithm
    alg = 'subpaving';

    % Param struct containing the system parameter
    % .U set of input constraints
    Param.U = interval(-max_torque, max_torque);
    % .W set of disturbances W
    Param.W = interval(zeros(2,1));
    % .V set of measurement errors
    % .X set of state constraints
    Param.X = mptPolytope([-pi -pi pi pi; -max_thdot max_thdot max_thdot -max_thdot]');

    % Opts struct containing algorithm settings
    Opts.Q = eye(2); %Default identity matrix
    Opts.R = 1; %Default all-zero matrix

    % Search domain
    % Chosen bigger than constrains allow on purpose
    vecTdomain = [1.5*pi; 5*pi];
    Opts.Tdomain = interval(-vecTdomain,vecTdomain);

    % Initial guess
    vecTinit = 0.1 * ones(2,1);
    Opts.Tinit = interval(-vecTinit,vecTinit);

    % Equilibrium point
    Opts.xEq = zeros(2,1);
    Opts.uEq = 0;

    % Recursion limit
    Opts.numRef = 6; %Default 4

    % Enlargement factor
    Opts.enlargeFac = 1.1; %Default 1.5

    % Final time for reachability analysis
    Opts.tMax = 100;

    % Time step size for reachability analysis.
    Opts.timeStep = 0.05;

    % Struct containing the settings for reachability analysis(CORA)
    %.cora

    % Compute terminal region
    clock = tic;
    T = computeTerminalRegion(f, alg, Param, Opts);
    tComp = toc(clock);

    disp([newline,'Computation time: ',num2str(tComp),'s', newline]);

    % Subpaving boxes (Box, Inf/Max, x/y)
    b = zeros(size(T.subpaving, 2), 2, 2);
    for i = 1:length(T.subpaving)
        b(i,:,:) = [infimum(T.subpaving{i}).'; supremum(T.subpaving{i}).'];
    end

    %Polytope vertices
    v = vertices(T.set);

    % Simulate terminal region controller
    tFinal = Opts.tMax;
    res = simulateRandom(T,tFinal,100,0.1,0.6,10);

    % Visualization
    figure; hold on; box on;
    for i = 1:length(T.subpaving)
       plot(T.subpaving{i},[1,2],'r');
    end
    plot(T.set,[1,2],'b');
    xlabel('x_1'); ylabel('x_2');

    %figure; hold on; box on;
    %plot(T.set,[1,2],'b');
    %plotSimulation(res,[1,2],'k');
    %xlabel('x_1'); ylabel('x_2');

end
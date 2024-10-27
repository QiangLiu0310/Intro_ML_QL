close all; clear; clc;

true_position = [0.0, 0.0]; % True position within unit circle
sigma_r = 0.30;       
sigma_x = 0.25;  
sigma_y = 0.25;           
K_values = [1:4];    

% Define grid for contour plot
x_range = -2:0.05:2;
y_range = -2:0.05:2;
[X, Y] = meshgrid(x_range, y_range);

figure;
for idx = 1:length(K_values)
    K = K_values(idx);

    % Generate landmarks on unit circle
    theta = linspace(0, 2*pi, K+1);
    theta(end) = []; 
    landmarks = [cos(theta)', sin(theta)'];

    true_ranges = vecnorm(landmarks - true_position, 2, 2); 
    noisy_ranges = true_ranges + sigma_r * randn(K, 1);   

    % Evaluate MAP objective function over grid
    Z = arrayfun(@(x, y) map_objective([x, y], landmarks, noisy_ranges, sigma_r, sigma_x, sigma_y), X, Y);

    % Plot contour map
    subplot(2, 2, idx);
    contourf(X, Y, Z, 20, 'LineColor', 'none'); % 20 contour levels
    colorbar;
    hold on;

    % Plot true position and landmarks
    plot(true_position(1), true_position(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    scatter(landmarks(:,1), landmarks(:,2), 80, 'yellow', 'filled', 'MarkerEdgeColor', 'black');
    % Titles and labels
    title(sprintf('K = %d Landmarks', K));
    xlabel('x');
    ylabel('y');
    axis equal;
    xlim([-2, 2]);
    ylim([-2, 2]);
    hold off;
end
sgtitle('MAP Objective Function Contours for Different K Values');

function cost = map_objective(pos, landmarks, noisy_ranges, sigma_r, sigma_x, sigma_y)
% pos: [x, y] candidate position
x = pos(1);
y = pos(2);
prior_cost = (x^2) / (2 * sigma_x^2) + (y^2) / (2 * sigma_y^2);
distances = vecnorm(landmarks - pos, 2, 2); % Euclidean distances to landmarks
measurement_cost = sum((noisy_ranges - distances).^2) / (2 * sigma_r^2);
cost = prior_cost + measurement_cost; % Total cost (negative log-posterior)
end

function U_struct = split_controls(U)
    U_struct = struct;
    U_struct.throttle = U(1,:);
    U_struct.aileron = rad2deg(U(2,:));
    U_struct.elevator = rad2deg(U(3,:));
    U_struct.rudder = rad2deg(U(4,:));
end


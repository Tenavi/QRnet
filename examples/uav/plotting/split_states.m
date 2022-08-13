function X_struct = split_states(t, X)
    X_struct = struct;
    X_struct.h = -X(1,:);
    X_struct.u = X(2,:);
    X_struct.v = X(3,:);
    X_struct.w = X(4,:);
    X_struct.p = rad2deg(X(5,:));
    X_struct.q = rad2deg(X(6,:));
    X_struct.r = rad2deg(X(7,:));
    X_struct.quaternion = X(8:end,:);
    [yaw, pitch, roll] = quat2eul123(X_struct.quaternion);
    X_struct.yaw = rad2deg(yaw);
    X_struct.pitch = rad2deg(pitch);
    X_struct.roll = rad2deg(roll);
    
    X_struct.Va = sqrt(X_struct.u.^2 + X_struct.v.^2 + X_struct.w.^2);
    X_struct.alpha = rad2deg(atan2(X_struct.w, X_struct.u));
    X_struct.beta = asin(X_struct.v ./ X_struct.Va);
    
    q0 = X(end,:);
    q1 = X(8,:);
    q2 = X(9,:);
    q3 = X(10,:);

    nt = size(X,2);
    pn_dot = zeros(1,nt);
    pe_dot = zeros(1,nt);
    for k=1:nt
        [pn_dot(k), pe_dot(k)] = pos_dynamics(...
            q0(k), q1(k), q2(k), q3(k),...
            X_struct.u(k), X_struct.v(k), X_struct.w(k));
    end
    
    if nt == 1
        X_struct.pn = 0;
        X_struct.pe = 0;
    else
        [X_struct.pn, X_struct.pe] = get_positions(t,pn_dot,pe_dot);
    end
    
    X_struct.course = rad2deg(atan2(pe_dot, pn_dot));
end


function [pn,pe] = get_positions(t,pn_dot,pe_dot)
    pn_dot = spline(t,pn_dot);
    pe_dot = spline(t,pe_dot);

    [~,pn] = ode45(@(t,pn) ppval(pn_dot,t), t, 0);
    [~,pe] = ode45(@(t,pe) ppval(pe_dot,t), t, 0);
end


function [yaw, pitch, roll] = quat2eul123(q)
    % Converts quaternions to Euler angles for the rotation
    % sequence 123.
    q0 = q(4,:);
    q1 = q(1,:);
    q2 = q(2,:);
    q3 = q(3,:);
    roll = atan2(2*(q0.*q1 + q2.*q3),...
                q3.^2 - q2.^2 - q1.^2 + q0.^2);
    pitch = asin(2*(q0.*q2 - q1.*q3));
    yaw = atan2(2*(q1.*q2 + q0.*q3),...
                q0.^2 + q1.^2 - q2.^2 - q3.^2);
end

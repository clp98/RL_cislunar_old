%-------------------------------------------------------------------------%
function [] = PoFileCutter(poFile, everyOtherLine)
%-------------------------------------------------------------------------%

%{

Author: Kristofer Drozd, Luca Ghilardi, Andrea Scorsoglio
Created: 03/02/2021

Description:
This functions cuts down a periodic orbit file

Input: 
poFile - Periodc orbit file name.
everyOtherLine - skip every # line.

Output: 

Version:
- 03/02/2021 - Initial development completed.



     ."".    ."",
     |  |   /  /
     |  |  /  /
     |  | /  /
     |  |/  ;-._
     }  ` _/  / ;
     |  /` ) /  /
     | /  /_/\_/\
     |/  /      |
     (  ' \ '-  |
      \    `.  /
       |      |
       |      |

%}

fid1 = fopen(poFile);
fid2 = fopen('cut_po_gile.txt','w');

n = 0;
tline = fgetl(fid1);

while ischar(tline)
    tline = fgetl(fid1);
    if ischar(tline) 
        if rem(n, everyOtherLine) == 0
            fprintf(fid2, '%s\n' ,tline);
        end
        n = n+1;
    end
end

fclose(fid1);
fclose(fid2);


% Plotting orbits for reference
close all
System = CrtbpEarthMoonInitializationParamaters();
orb_IC = load('cut_po_gile.txt');
orb = PropagateOrbitFamily(System, orb_IC, 0.01, 2);

figure(1); hold on; grid on; axis equal
plot3(-System.mu,0,0,'k.','markersize',20)
plot3(1-System.mu,0,0,'g.','markersize',20)
for i=1:orb.nOrb
    plot3(orb.states{i,1}(:,1),orb.states{i,1}(:,2),orb.states{i,1}(:,3))
end


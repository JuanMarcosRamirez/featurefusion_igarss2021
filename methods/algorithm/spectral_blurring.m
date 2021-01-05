function [I_MS] = spectral_blurring(Io, q, dec_option)
% spectral_blurring returns a spectrally downsampled version of the input
% spectral image, with a spectral decimation factor q. The spectral image
% is downsampled according to the strategy specified in the dec_option
% input.
%
% [I_MS] = spectral_blurring(Io, q, dec_option)
%
%   Inputs:
%   I               = input spectral image
%   q               = spectral decimation factor
%   dec_option:
%
%   'decimation':Selects a set of spectral bands of the input image,
%   i.e, it extracts each band directly from the input image Io, at regular
%   intervals determined by q (default).
%
%   'average'   :Each band of the output image is obtained as the
%   average of a set of spectral bands of the input image Io.
%
%   Outputs: I_MS    = spectrally downsampled version of the input
%   spectral image
%
%   Reference: 
%
%   [1] Juan Marcos Ramirez and Henry Arguello, "Spectral Image
%   Classification From Data Fusion Compressive Measurements"
%
%   Authors:
%   Juan Marcos Ramirez.
%   Universidad Industrial de Santander, Bucaramanga, Colombia
%   email: juanmarcos26@gmail.com
%
%   Date:
%   May, 2018
%
%   Copyright 2018 Juan Marcos Ramirez Rondon.  [juanmarcos26-at-gmail.com]

%   This program is free software; you can redistribute it and/or modify it
%   under the terms of the GNU General Public License as published by the
%   Free Software Foundation; either version 2 of the License, or (at your
%   option) any later version.
% 
%   This program is distributed in the hope that it will be useful, but
%   WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%   General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License along
%   with this program; if not, write to the Free Software Foundation, Inc.,
%   675 Mass Ave, Cambridge, MA 02139, USA.


if (nargin == 2)
    dec_option = 'decimation';
end

[M,N,L] = size(Io);
I_MS = zeros(M,N,floor(L/q));

if strcmp(dec_option,'decimation')
    for i = 1:floor(L/q)
        I_MS(:,:,i)   = Io(:,:,(i-1)*q + 1);
    end
elseif strcmp(dec_option, 'average')
    for i = 1:floor(L/q)
        I_MS(:,:,i)   = mean(Io(:,:,(i-1)*q + 1:i*q),3);
    end
elseif strcmp(dec_option, 'sum')
    for i = 1:floor(L/q)
        I_MS(:,:,i)   = sum(Io(:,:,(i-1)*q + 1:i*q),3);
    end
end
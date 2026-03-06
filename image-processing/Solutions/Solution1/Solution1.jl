### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 649dbe1a-2da5-11ec-3f4e-39b5d6dfe737
begin
	using PlutoUI, Images,TestImages, CairoMakie, LinearAlgebra, Interpolations, ImageTransformations,CoordinateTransformations, Rotations, StaticArrays

	PlutoUI.TableOfContents(depth=6)
end

# ╔═╡ 27d1f36f-6329-4030-b7a9-f8ee0630ffe8
begin
	include("./Img/shrek_int.jl");
	testimg = load("./Img/testimg.jpg")
	shrekRGB = load("./Img/shrek.jpg")
end

# ╔═╡ 3e968320-e085-4e67-9111-f1d4b3292745
md"""
# Image Processing - Exercise 1
[Institute for Biomedical Imaging](https://www.tuhh.de/ibi/home.html), Hamburg University of Technology

* 👨‍🏫 Prof. Dr.-Ing. Tobias Knopp, 
* 🧑‍🏫 [Sarah Reiß, M.Sc.](mailto:sarah.reiss@tuhh.de)

📅 Due date: 28.10.2024, 1 pm
"""

# ╔═╡ d5e0b830-34fc-4666-8e71-71cef73f2cba
md"""
**Exercise Procedure**

The weekly exercises will be handed out every Tuesday afternoon and need to be handed in until the following Tuesday, 1 pm. The correction of the programming tasks will be done by autograding. Keep that in mind when you alter the given function-bodies. All tasks are denoted by a 🎓-sign. Hard tasks will be denoted by two 🎓🎓-signs. Sometimes there are (proofing-) exercises, that you should solve separetely on a sheet of paper. These exercises will not be corrected but are also important as a preparation for the exam.
"""

# ╔═╡ 7d88ae11-db3c-4b4e-819d-dddfccd61a46
md"""
### 1. Digital Images
For a general overview on how to work with images in julia consider this [link](https://juliaimages.org/stable/tutorials/quickstart/).

In this exercise we will use an image of Shrek. The first challenge is to convert the RGB-Image to a grayscaled image using our own functions.
"""

# ╔═╡ c6de2614-04b5-4030-a814-835dbf5bc02d
md"""
There are different ways to achieve this conversion. The first and most simple way is to take the mean value of the red-, green- and blue-values for each pixel.
###### 🎓 a) 
**Write a function `RGB2G_mean()`, which converts an arbitrary RGB-Image to a grayscaled Image by calculating the mean value for each pixel. Hint: use broadcasting on the julia-functions red(), green() and blue().**
"""

# ╔═╡ 6c23560b-010f-4034-88e9-11625514bf15
function RGB2G_mean(img::Matrix{RGB{T}}) where T<:Fractional
	R,G,B = Float64.(red.(img)),Float64.(green.(img)),Float64.(blue.(img))
	return Gray.((R+G+B)/3)
end

# ╔═╡ e3d8cf3f-abe6-4e56-8c90-d6ed87188376
md"""
The Problem with this simple conversion is, that the perceptional lightness our eye detects is not the same for different colors. To overcome this, one can weight the RGB-values to get a better total luminance. Luckily there have been some studies to find good values for the luminance-weighting factors and we can use them.
###### 🎓 b) 
**Write a function `RGB2G_corrected()`, which converts an arbitrary RGB-Image to a grayscaled Image by using the following weighting factors**

$I_{G}=0.2126I_{R}+0.7152I_{G}+0.0722I_{B}.$
"""

# ╔═╡ bf86aa2d-ff6e-4063-b9eb-bb43813c2380
function RGB2G_corrected(img::Matrix{RGB{T}}) where T<:Fractional
	R,G,B = Float64.(red.(img)),Float64.(green.(img)),Float64.(blue.(img))
	return Gray.(0.2126*R+0.7152*G+0.0722*B)
end

# ╔═╡ 516cc797-aa1c-4c53-a672-2c7c41077434
md"""
There is still another nonlinearity when it comes to the perceptional lightness. The human eye is very sensitive to changes in low luminance but less sensitiv for changes in high luminance. To overcome this, we need to convert the nonlinear RGB-values with the so called gamma expansion to linear R'G'B' values:

$C_{lin}=\begin{cases}
	\frac{C}{12.91},	& \text{if} ~C\leq 0.04045 \\
	\left(\frac{C+0.055}{1.055}\right)^{2.4},	& \text{otherwise}.
\end{cases} \text{ for } C \in \{R,G,B\}$
After that we can adjust the luminance of the different colors to get the linear grayscaled image.
And as a last step we need to perform a inverse gamma expansion to get the nonlinear grayscaled image:

$G =\begin{cases}
	12.91 G_{lin},	& \text{if} ~G_{lin}\leq 0.0031308 \\
	1.055 G_{lin}^{1/2.4} -0.055,	& \text{otherwise}.
\end{cases}$
###### 🎓 c) 
**Write a function `RGB2G_gammaCompression()`, which converts an arbitrary RGB-Image to a grayscaled Image by using the above weights, the gamma expansion and the inverse gamma expansion.**

"""

# ╔═╡ 2555fea0-162c-4d36-8a50-80f609aa0399
function RGB2G_gammaCompression(img::Matrix{RGB{T}}) where T<:Fractional
	R,G,B = Float64.(red.(img)),Float64.(green.(img)),Float64.(blue.(img))
	for (j,A) in enumerate([R,G,B])	
		for i in eachindex(A)
			if A[i]<=0.04045
				A[i]=A[i]/12.91
			else
				A[i]=((A[i]+0.055)/1.055)^2.4
			end
		end
	end
	Y=0.2126*R+0.7152*G+0.0722*B
	for i in eachindex(Y)
		if Y[i]<=0.0031308
			Y[i]=Y[i]*12.91
		else
			Y[i]=(Y[i])^(1/2.4)*1.055-0.055
		end
	end	
	return Gray.(Y)
end

# ╔═╡ ea5498b1-90b8-47d0-8e8a-d40e77c0dbab
md"""
While the last method gives a really good result, it has a much higher computationally cost in comparison with a simple weighting. Therefore, for modern color spaces (as well as in the Images.jl converting function) one uses a linear approximation of the gamma compression method and ends up with the weighting factors

$I_{G}=0.299I_{R}+0.587I_{G}+0.114I_{B}.$
###### 🎓 d) 
**Copy and adjust the function from b) to a function `RGB2G_luna()`, which converts an arbitrary RGB-image to a grayscaled Image by using the above given new weighting factors.**
"""

# ╔═╡ f100b3b5-45c1-44b2-8f86-9da0622a5ebd
function RGB2G_luma(img::Matrix{RGB{T}}) where T<:Fractional
	R,G,B = Float64.(red.(img)),Float64.(green.(img)),Float64.(blue.(img))
	return Gray.(0.299*R+0.587*G+0.114*B)
end

# ╔═╡ fff1ecce-1f11-4b56-b0ba-be895891c8e5
let
	f = Figure()
	ax1=image(f[1,1],rotr90(RGB2G_mean(shrekRGB)),axis = (title = "mean",aspect = 1))
	hidedecorations!(ax1.axis)
	
	ax2=image(f[1,2],rotr90(RGB2G_corrected(shrekRGB)),axis = (title = "corrected",aspect = 1))
	hidedecorations!(ax2.axis)
	
	ax3=image(f[2,1],rotr90(RGB2G_gammaCompression(shrekRGB)),axis = (title = "gamma compression",aspect = 1))
	hidedecorations!(ax3.axis)
	
	ax4=image(f[2,2],rotr90(RGB2G_luma(shrekRGB)),axis = (title = "linear approximation",aspect = 1))
	hidedecorations!(ax4.axis)
	
	f
end

# ╔═╡ 306971e0-d544-4e17-86f8-021151c30445
md"""
Now we want to reformulate our grayscaled float-based Image to an integer-image. The basic procedure to gain an integer image is considered in the lecture.
###### 🎓 e)
**Write a function `float2int()`, which converts a matrix of Floats to a matrix of UInt8. If the input matrix is constant in all entries (!division by zero..!) return a zero-matrix of type UInt8.**
"""

# ╔═╡ 280315a1-5831-4e6f-b3bd-387e85e8f8fe
function float2int(A::Matrix{T}) where T<:Fractional
	β=minimum(A)
	α=maximum(A)-β
	if α==0
		return zeros(UInt8,size(A))
	else
		A_norm=(A.-β)./α
		return UInt8.(round.((2^8-1)*A_norm))
	end
end

# ╔═╡ b95bbc90-24c7-4c57-906f-b82efad06ca1
begin
	a1=heatmap(rotr90(shrek_int),colormap="grays",axis=(;aspect=1))
	hidedecorations!(a1.axis)
	a1
end

# ╔═╡ aed1bc0a-0bf6-4fda-8959-2cb021d1a8ad
md"""
If we consider the binary coefficients of the different bit levels of the UInt8-image, we can see how the complete image is composed:

$I_{i,j}=\sum_{k=1}^{8} b_k\cdot 2^{k-1}, \text{ where } b_i\in\{0,1\}.$

Cleary the coefficient $b_8$ has the most impact on the image and is therefore called the "most significant bit".
###### 🎓 f)
**Use the Julia function `digits()` on the array `shrek_int` to get the binary composition of each pixel of our UInt8-Image. Store the 8 bit levels of the picture in an 8x256x256 Boolean-array `bitlevels_shrek`.**
"""

# ╔═╡ 9f6dcc3d-b3dc-400d-8d69-44e82cf7b6e1
begin
	bits=digits.(shrek_int,base=2,pad=8)
	bitlevels_shrek=Array{Bool}(undef,8,256,256)
	for k=1:8
		for i=1:256
			for j=1:256
				bitlevels_shrek[k,j,i]=bits[j,i][k]	
			end
		end
	end
end

# ╔═╡ fda0e2ca-664c-4a66-96a5-38f5c7d22499
let
	f=Figure()
	ax=[]
	for i=1:2, j=1:4
		
		k=(i-1)*4+j
		
		push!(ax,image(f[i,j],rotr90(Gray.(bitlevels_shrek[k,:,:])),axis = (title = "bit level $k",aspect = 1)))
	
		hidedecorations!(ax[k].axis)
	
	end
	f
end

# ╔═╡ 82dca013-560c-4bee-a894-71b6ed4ca5c7
md"""
The first two bit-levels show normaly a nearly random distribution. Humans can distinguish up to 30 shades of gray in an image, and one only needs 5 bits to accomplish this. The lower (less significant) bit-levels can therefore be used to hide messages in images (steganography), which can't be seen by humans in the composed image. Whereas if one places messages in the mid bit-levels they appear as watermarks in the composed image.
"""

# ╔═╡ b80ae2d8-5574-49dc-92cb-063df31d2b89
md"""
### 2. Interpolation
We want to implement some of the interpolation-methods discussed in the lecture.
The most simple one is the Nearest-Neighbor-Interpolation. The procedure in 1D is as follows:
* Input: xs: vector of gridpoints, v: vector of values on the gridpoints, x: position on which we want to calculate an interpolated value
* Check that xs and v have the same length. Check if x lies directly on a gridpoint. If so: find this gridpoint and return the value on this gridpoint. Else:
* Calculate the distance from the given position x to the next grid points on the left and on the right.
* If the left distance is smaller or equal than the right distance return the value from the left grid point, otherwise return the value from the right gridpoint.
###### 🎓 a)
**Implement the function `NN_1D`, which takes a vector of positions `xs`, a vector of values `v` and a position `x` and returns the Nearest-Neighbor-Interpolation on the postition `x`. Make sure that the function also terminates for the edges and returns a reasonable value. Hint: you may find the julia-functions `findfirst()` and/or `findlast()` useful.**
"""

# ╔═╡ af2cf3fd-bdf0-45b3-8d87-4f828f87fd1f
function NN_1D(xs,v,x)
	@assert length(xs)==length(v)
	if x in xs
		return v[findfirst(z->z==x,xs)] 
	else		
		xl=findlast(z->(z<=x),xs)
		norm(xs[xl]-x)<=norm(xs[xl+1]-x) ? (return v[xl]) : (return v[xl+1])
	end
end

# ╔═╡ de91fefd-229e-45a1-8573-11570228d428
md"""
Now for the procedure in 2D. We can make use of the 1D interpolation, since a 2D-NN-interpolation is nothing more than a composition of two 1D-NN-interpolations: 
* Input: xs: vector of gridpoints in first dimension, ys: vector of gridpoints in second dimension, A: matrix of values on the gridpoints, x: 2D-position on which we want to calculate an interpolated value
* Start with the first dimension: if the position in this dimension directly lies on a gridpoint in this dimension, return the 1D-NN-interpolation for the second dimension (the input-matrix becomes a vector by fixing the position in the first dimension). Else:
* Calculate the distance in the first dimension from the given position x to the next grid points on the left and on the right.
* If the left distance is smaller or equal than the right distance return the 1D-NN-interpolation with the input matrix fixed in the first dimension on the left grid point, otherwise return the 1D-NN-interpolation with the input matrix fixed in the first dimension on the right gridpoint.

###### 🎓 b)
**Implement the function `NN_2D`, which takes two vectors of positions `xs` and `ys`, a matrix of values `v` and a 2D-position `x=(x[1],x[2])` and returns the Nearest-Neighbor-Interpolation on the postition `x=(x[1],x[2])`.**
"""

# ╔═╡ 51587457-5b8a-4c5a-89a9-dbf9fd9ec5ef
function NN_2D(xs,ys,A,x)
	@assert length(xs)==size(A)[1]
	@assert length(ys)==size(A)[2]
	
	if x[2] in ys
		return NN_1D(xs,A[:,findfirst(z->z==x[2],ys)],x[1])
	else
		yl=findlast(z->(z<=x[2]),ys)
		norm(ys[yl]-x[2])<=norm(ys[yl+1]-x[2]) ? vy=yl : vy=yl+1
		return NN_1D(xs,A[:,vy],x[1])
	end
end

# ╔═╡ 1f1d5598-a26f-4875-8642-ec20750dd252
md"""
Now we want to switch to a more advanced interpolation. The basic procedure of a linear interpolation in 1D is as follows: 
* Input: xs: vector of gridpoints, v: vector of values on the gridpoints, x: position on which we want to calculate an interpolated value
* Check that xs and v have the same length. Check if x lies directly on a gridpoint. If so: find this gridpoint and return the value on this gridpoint. Else:
* Calculate the distance from the given position x to the next grid point on the left.
* Divide it by the distance between both neighbor-gridpoints of x and save the result as a weighting factor w.
* Return the sum of the linearly weighted values of the left grid point (by 1-w) and the right grid point (by w).

###### 🎓 c)
**Implement the function `lin_1D`, which takes a vector of positions `xs`, a vector of values `v` and a position `x` and returns the linear interpolation on the postition `x`. Again make sure, that the function also terminates for values on the edge.**
"""

# ╔═╡ 37dde329-0ef5-4e77-a6b4-e62b0ce425ff
function lin_1D(xs,v,x)
	@assert length(xs)==length(v)
	if x in xs
		return v[findfirst(z->z==x,xs)] 
	else
		xl=findlast(z->(z<=x),xs)
		w=norm(xs[xl]-x)/norm(xs[xl]-xs[xl+1])
		return v[xl[end]]*(1-w)+v[xl[end]+1]*w
	end
end

# ╔═╡ 078dc45c-97ea-408d-af4b-954f41edf53f
md"""
Now for the 2D-linear-interpolation. Again we can make use of the 1D-linear- interpolation. The procedure is similar to the 2D-NN-interpolation.
###### 🎓🎓 d)
**Implement the function `lin_2D`, which takes a vector of positions `xs`, a vector of values `v` and a 2D-position `x` and returns the linear interpolation on the postition `x`.**
"""

# ╔═╡ 3b59e67e-4f24-4eeb-84a1-a22c5d4a32ad
function lin_2D(xs,ys,A,x) 
	@assert length(xs)==size(A)[1]
	@assert length(ys)==size(A)[2]
	
	if x[2] in ys
		return lin_1D(xs,A[:,findfirst(z->z==x[2],ys)],x[1])
	else
		yl=findlast(z->(z<=x[2]),ys)
		w=norm(ys[yl]-x[2])/norm(ys[yl]-ys[yl+1])
		return lin_1D(xs,A[:,yl],x[1])*(1-w)+lin_1D(xs,A[:,yl+1],x[1])*w
	end
end

# ╔═╡ b27a908f-eb2d-47ed-b2eb-f9b7418b82c1
begin
	f_s=Figure()
	shrekGray=Gray.(shrekRGB)
	
	badshrek=shrekGray[35:115,100:180];	badshrek[2:2:80,:].=0; badshrek[:,2:2:80].=0
	
	ax1 = image(f_s[1,1],rotr90(Gray.(shrekGray[35:115,100:180])),axis=(title="original Shrek",aspect=1))
	hidedecorations!(ax1.axis)
	
	ax2 = image(f_s[1,2],rotr90(Gray.(badshrek)),axis=(title="missing-data Shrek",aspect=1))
	hidedecorations!(ax2.axis)

	smallShrek = shrekGray[35:2:115,100:2:180]
	
	NN_shrek=[NN_2D(collect(35:2.0:115),collect(100:2.0:180),smallShrek,(x,y))
		for x in 35:1.0:115, y in 100:1.0:180]

	ax3 = image(f_s[2,1],rotr90(Gray.(NN_shrek)),axis=(title="NearestNeighbor Shrek",aspect=1))
	hidedecorations!(ax3.axis)

	lin_shrek=[lin_2D(collect(35:2.0:115),collect(100:2.0:180),smallShrek,(x,y))
		for x in 35:1.0:115, y in 100:1.0:180]
	ax4 = image(f_s[2,2],rotr90(Gray.(lin_shrek)),axis=(title="linear Shrek",aspect=1))
	hidedecorations!(ax4.axis)

	f_s
end

# ╔═╡ 9bf6df9d-bdeb-461d-b4f4-669e29136470
md"""
### 3. Geometric Transformations - Rotation

The next goal is to make use of our interpolation routines to implement an image-rotation.
As you learned in the lecture, we can perform a rotation through a affine linear transformation

$\varphi(\textbf{r}) = \textbf{A} \textbf{r} + \textbf{b},$

where

$\textbf{A}=\begin{pmatrix}\cos \theta& \sin \theta\\-\sin \theta& \cos \theta \end{pmatrix}\text{ and }\textbf{b}=\begin{pmatrix}0\\0 \end{pmatrix}.$

After we performed the coordinate-transformation on our grid, the new rotated gridpoints may not lay directly on the old gridpoints and we have to perform an interpolation.

We solve the problem in 2 steps. In the first step we implement a helper-function which performs the interpolation. If you did not solve task 2 successfully you may use the julia interpolation routines from Interpolations.jl.
In the second step we implement the rotation-function itself.

###### 🎓 a)
**Write a function `isinside(x,img,linear)` which takes a rotated 2D-coordinate `x`, a grayscaled image `img` and a boolean variable `linear`. The function should decide, whether the coordinate is inside the "old" grid of `img`. If it is not, the function should return 0, else it should return the interpolated value of `img` at the position `x`. If `linear` is true it should use a linear interpolation, if not it should use a NN-interpolation.**
"""

# ╔═╡ 645cc476-85df-4bb3-b593-b367911f98cb
function isinside(x,img;linear=true)
	N=size(img)
	if 1.0<=x[1]<=N[1] && 1.0<=x[2]<=N[2]
		if linear
			lin_2D(1:N[1],1:N[2],img,x)
		else
			NN_2D(1:N[1],1:N[2],img,x)
		end
	else
		return 0
	end
end

# ╔═╡ 98d5a3a6-233e-4902-89f3-c0592c9e828d
md"""
Now for the rotation itself. The main part is to generate the rotated grid. For this reason we need to read out the "old" grid size $N=(N[1],N[2])$ of the Image and calculate how large the rotated image will be. This is a simple geometric calculation: the new grid size is given by $\text{Int}(\text{round}(\sin(\theta)*N[1]+\cos(\theta)*N[2])) \text{ for } \theta \in [0,\frac{\pi}{2}]$.
We cannot multiply the rotation-matrix directly to the gridpoints, because we want a rotation around the center of the image. Therefore we need to recenter the origin before the rotation and again after the rotation. One usually calculates the rotated grid points and looks via an inverse rotation, which value the old gridpoints had.

###### 🎓🎓 b)
**Write a function `myrotation(img,Θ;method)` which takes a grayscaled image `img`, an angle `Θ` and a boolean variable `linear` and returns a clockwise rotation by `Θ` of the image. Use the function `isinside()` from a) for the interpolation-step, where the boolean `linear` decides if the interpolation is linear or NN.**
"""

# ╔═╡ 6bae3241-79d4-475d-8989-c8bfd541debc
function myrotation(img,Θ;linear=true)
	N = size(img)
	M = Int(round(sin(Θ)*N[1]+cos(Θ)*N[2]))
	A = inv([cos(Θ) sin(Θ); # RotMatrix(Θ)
			-sin(Θ) cos(Θ)])
	rotInds=[A*([x;y].-((M.+1)./2)).+((N.+1)./2) for x in 1:M,y in 1:M]
		
	return Gray.(isinside.(rotInds,(img,);linear=linear))
end

# ╔═╡ 50d41a88-ed7d-486a-8af9-7f93eac51ffd
let
	f=Figure()
	
	prNN=image(f[1,1],rotr90(myrotation(shrekGray,pi/4;linear=false)),axis=(title="NearestNeighbor",aspect=1))
	hidedecorations!(prNN.axis)
	
	prlin=image(f[1,2],rotr90(myrotation(shrekGray,pi/4;linear=true)),axis=(title="linear",aspect=1))
	hidedecorations!(prlin.axis)
	f
end

# ╔═╡ f93d09f8-f10d-4d13-baa4-9593dcf31531
md"""
### 4. Bonus: Shearing
Now we want to implement shearing. As you may recall from the lecture, the transformation-matrix $\textbf{A}$ is either given by 

$\begin{pmatrix}1&s_v\\0&1 \end{pmatrix} \text{ or } \begin{pmatrix}1&0\\s_h&1 \end{pmatrix}.$

The basic functionality is the same as for the rotation. Again, we need to consider that we want to shear around the image center. The main difference is the calculation of the size of the transformed grid, which depends on sv and sh.

###### 🎓 c) 
**Write a function `myshearing(img,sv,sh;linear)` which takes a grayscaled image `img`, the shearing parameters `sv` and `sh` and the boolean variable `linear` and returns the sheared image. Make use of the function `isinside()` again.**
"""

# ╔═╡ f55360b2-03bd-41d1-85d3-353bef0b4d52
function myshearing(img,sv,sh;linear=true)
	N=size(img)
	M=Int.(round.((N[1]+abs(sv)*(N[2]-1),N[2]+abs(sh)*(N[1]-1))))
	A = inv([1 sv;
		 	 sh 1])
	shearInds=[A*([x,y].-((M.+1)./2)).+((N.+1)./2) for x in 1:M[1], y in 1:M[2]]
	return Gray.(isinside.(shearInds,(img,);linear=linear))
end

# ╔═╡ 993059bf-0da5-45f3-a59a-ed40d72c993d
let
	f=Figure()
	
	psNN=image(f[1,1],rotr90(myshearing(shrekGray,0.0,1.0;linear=false)),axis=(title="NearestNeighbor",aspect=1))
	hidedecorations!(psNN.axis)
	
	pslin=image(f[1,2],rotr90(myshearing(shrekGray,0,1;linear=true)),axis=(title="linear",aspect=1))
	hidedecorations!(pslin.axis)
	f
end

# ╔═╡ cc609e7a-d56d-4977-a433-d9a30ecbf4d2
md"""
A cool thing about shearing is that one can formulate a rotation as a conjugation of three shearing transformations (in fact one can formulate each orthogonal transformation as a conjugation of three shearing transformations). 

###### 🎓 d) 
**Assume that you have a conjugation of three shearing transformations of the form $\begin{pmatrix}1&\alpha\\0&1 \end{pmatrix} \begin{pmatrix}1&0\\\beta&1 \end{pmatrix} \begin{pmatrix}1&\gamma\\0&1 \end{pmatrix}$. Show that a rotation is a conjugation of three shearing transformations by deriving the three transformation matrices and define the values of $\alpha$, $\beta$ and $\gamma$ for $\theta=\frac{\pi}{4}$.**

From

$\begin{pmatrix}1&\alpha\\0&1 \end{pmatrix} \begin{pmatrix}1&0\\\beta&1 \end{pmatrix} \begin{pmatrix}1&\gamma\\0&1 \end{pmatrix}=\begin{pmatrix}1+\alpha\beta&\alpha+\gamma+\alpha\beta\gamma\\\beta&1+\beta\gamma \end{pmatrix}\overset{!}{=}
\begin{pmatrix}\cos \theta& \sin \theta\\-\sin \theta& \cos \theta \end{pmatrix}$

it follows that

$\beta=-\sin\theta,$
$\alpha=\gamma=\frac{\cos\theta-1}{\beta}=\tan\frac{\theta}{2}.$

"""

# ╔═╡ 2c0613ee-bfb6-476e-97a3-7cab7df902df
begin
	Θ=pi/4
	α=tan(Θ/2)
	β=-sin(Θ)
	γ=tan(Θ/2)
end;

# ╔═╡ 0fd17c6a-5eb3-4b19-88f1-a509265c0e4d
begin
	prs1=myshearing(shrekGray,γ,0)
	prs2=myshearing(prs1,0,β)
	prs3=myshearing(prs2,α,0)
	
	f_sr=Figure()

	asr_1 = image(f_sr[1,1],rotr90(prs1),axis=(title="1.shearing",aspect=1))
	hidedecorations!(asr_1.axis)
	
	asr_2 = image(f_sr[1,2],rotr90(prs2),axis=(title="2.shearing",aspect=1))
	hidedecorations!(asr_2.axis)

	asr_3 = image(f_sr[1,3],rotr90(prs3),axis=(title="3.shearing",aspect=1))
	hidedecorations!(asr_3.axis)
	
	f_sr
end

# ╔═╡ 448cb1b3-8fc0-425e-95aa-7eaa4c4a9acd
begin
	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))
	not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oh, oh! 😱", [md"Variable **$(Markdown.Code(string(variable_name)))** is not defined. You should probably do something about this."]))
	still_missing(text=md"Replace `missing` with your solution.") = Markdown.MD(Markdown.Admonition("warning", "Let's go!", [text]))
	keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]));
	yays = [md"Great! 🥳", md"Correct! 👏", md"Tada! 🎉"]
	correct(text=md"$(rand(yays)) Let's move on to the next task.") = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))
end;

# ╔═╡ 7ad27580-fb5b-4908-9944-3b89607571b9
hint(md"Use either broadcasting on the functions `red()`, `green()` and `blue()` or the function `channelview()` to get the RGB-values of an image.")

# ╔═╡ 1a087ab6-46d2-4835-9ae3-b9c259ed84dd
let  x = [ 0.403922  0.445752  0.555556   0.295425  0.786928   0.482353
 0.517647  0.433987  0.0888889  0.679739  0.648366   0.640523
 0.379085  0.359477  0.256209   0.235294  0.620915   0.490196
 0.433987  0.147712  0.598693   0.626144  0.43268    0.491503
 0.150327  0.470588  0.537255   0.501961  0.465359   0.345098
 0.415686  0.237908  0.559477   0.518954  0.0823529  0.533333]
	if RGB2G_mean(zeros(RGB{Float64},6,6)) != zeros(Gray{Float64},6,6)
		keep_working(md"`RGB2G_mean` does not return the desired output.")
	elseif !isapprox(Float64.(RGB2G_mean(testimg)),x,atol=1e-5)
		keep_working(md"`RGB2G_mean` does not return the desired output.")
	else
		correct()
	end
end

# ╔═╡ 5898be63-30fd-42d6-8773-49bf53e48e07
let  x = [ 0.367782  0.295856  0.722285  0.103025  0.761228   0.62226
 0.749464  0.447203  0.129476  0.84438   0.771329   0.769288
 0.197224  0.693676  0.140789  0.164219  0.734351   0.287416
 0.760557  0.113347  0.672403  0.564448  0.3513     0.470028
 0.157155  0.623973  0.7049    0.648642  0.473102   0.628096
 0.425198  0.193711  0.766743  0.601078  0.0899733  0.423518]
	if RGB2G_corrected(zeros(RGB{Float64},6,6)) != zeros(Gray{Float64},6,6)
		keep_working(md"`RGB2G_corrected` does not return the desired output.")
	elseif !isapprox(Float64.(RGB2G_corrected(testimg)),x,atol=1e-5)
		keep_working(md"`RGB2G_corrected` does not return the desired output.")
	else
		correct()
	end
end

# ╔═╡ fe20e217-da71-472d-9334-bfebee8ce07a
let  x = [ 0.37395   0.344716  0.772121  0.213719  0.762545   0.652326
 0.818683  0.505678  0.136307  0.862153  0.783163   0.782674
 0.270983  0.825535  0.1808    0.265814  0.756713   0.475895
 0.842291  0.12102   0.681544  0.568161  0.50842    0.535045
 0.167709  0.659295  0.739447  0.666114  0.492611   0.70487
 0.445134  0.263438  0.794784  0.618849  0.0999313  0.440443]
	if RGB2G_gammaCompression(zeros(RGB{Float64},6,6)) != zeros(Gray{Float64},6,6)
		keep_working(md"`RGB2G_gammaCompression` does not return the desired output.")
	elseif !isapprox(Float64.(RGB2G_gammaCompression(testimg)),x,atol=1e-5)
		keep_working(md"`RGB2G_gammaCompression` does not return the desired output.")
	else
		correct()
	end
end

# ╔═╡ 1b7793cd-c2f4-4420-ba5b-e1286d667cba
let  x = [ 0.368043  0.341682  0.655878  0.145133  0.771655   0.572773
 0.666063  0.475863  0.116929  0.816447  0.752173   0.732894
 0.243169  0.576263  0.164788  0.207686  0.690153   0.375639
 0.665671  0.125424  0.667443  0.579584  0.416933   0.438745
 0.164216  0.569961  0.648078  0.623322  0.489749   0.543341
 0.404737  0.227616  0.729357  0.599769  0.0953294  0.442937]
	if RGB2G_luma(zeros(RGB{Float64},6,6)) != zeros(Gray{Float64},6,6)
		keep_working(md"`RGB2G_luma` does not return the desired output.")
	elseif !isapprox(Float64.(RGB2G_luma(testimg)),x,atol=1e-5)
		keep_working(md"`RGB2G_luma` does not return the desired output.")
	else
		correct()
	end
end

# ╔═╡ 3b4ae844-efa7-452a-aacb-c3a0b6fccc6e
let  x = [ 0x61  0x57  0xc6  0x12  0xf0  0xa9;
 0xca  0x86  0x08  0xff  0xe9  0xe2;
 0x35  0xaa  0x19  0x28  0xd3  0x64;
 0xca  0x0b  0xca  0xac  0x72  0x7a;
 0x19  0xa8  0xc3  0xbb  0x8c  0x9f;
 0x6d  0x2f  0xe1  0xb3  0x00  0x7b]
	if float2int(zeros(Float64,6,6)) != zeros(UInt8,6,6) || 
		float2int(ones(Float64,6,6)+ones(Float64,6,6)*0.1) != zeros(UInt8,6,6)
		keep_working(md"`RGB2G_luma` does not return the desired output for a constant matrix.")
	elseif !isapprox(float2int(Float64.(Gray.(testimg))),x,atol=1e-5)
		keep_working(md"`RGB2G_luma` does not return the desired output.")
	else
		correct()
	end
end

# ╔═╡ 5c89a853-59ec-484c-bd3f-bb3f5932bb08
let  x =  [0  0  0  0  0  0  0  0  0  0;
 0  0  1  1  0  0  0  0  0  0;
 0  0  0  0  0  0  0  0  0  0;
 0  0  0  0  0  0  0  0  0  0;
 0  0  0  0  0  0  0  0  0  1;
 0  0  0  0  0  0  0  0  1  1;
 0  0  0  0  0  0  1  1  0  0;
 0  0  0  1  1  1  1  0  0  0;
 0  1  1  1  1  0  0  0  0  0;
 0  1  1  1  1  0  0  0  1  1]
	if	bitlevels_shrek[8,1:10,1:10] != zeros(Bool,10,10) ||
		bitlevels_shrek[1,100:109,200:209] != x
		keep_working(md"`bitlevels_shrek` is not filled correctly.")
	else
		correct()
	end
end

# ╔═╡ 74a7342e-24ac-4a3c-81a5-73dab7f878b2
let  (xs,A) = (collect(1:0.2:5),[log(x) for x in 1:0.2:5])
	if NN_1D(xs,A,1)!=0.0 || NN_1D(xs,A,5) != 1.6094379124341003
		keep_working(md"`NN_1D` does not return a reasonable value at the edges.")
	elseif NN_1D(xs,A,1.49)!=log(1.4) || NN_1D(xs,A,1.51)!=log(1.6)
		keep_working(md"`NN_1D` does not return the correct NN-interpolation.")
	else
		correct()
	end
end

# ╔═╡ 65b29352-d6f4-4008-bc26-85c2abd293e6
let  (xs,A) = (collect(1:1.0:10),[x*y for x in 1:1.0:10,y in 1:1.0:10])
	if NN_2D(xs,xs,A,(1.0,1.0)) != 1.0 || NN_2D(xs,xs,A,(5.6,10.0)) != 60.0
		keep_working(md"`NN_2D` does not return a reasonable value at the edges.")
	elseif NN_2D(xs,xs,A,(5.4,5.4)) != 25.0 || NN_2D(xs,xs,A,(5.6,5.6)) != 36.0
		keep_working(md"`NN_2D` does not return the correct NN-interpolation.")
	else
		correct()
	end
end

# ╔═╡ 152c07d3-6c14-4ac4-bf16-1fb26b5b1dbd
let  (xs,A) = (collect(1:0.2:5),[log(x) for x in 1:0.2:5])
	if lin_1D(xs,A,1)!=0.0 || lin_1D(xs,A,5) != 1.6094379124341003
		keep_working(md"`lin_1D` does not return a reasonable value at the edges.")
	elseif !isapprox(lin_1D(xs,A,1.5),0.40323793293347426) ||
		   !isapprox(lin_1D(xs,A,2.42),0.8834730081212535)
		keep_working(md"`lin_1D` does not return the correct linear interpolation.")
	else
		correct()
	end
end

# ╔═╡ 4106e8e6-b414-4bf6-8f1c-89bd3af44a1d
let  (xs,A) = (collect(1:1.0:10),[x*y for x in 1:1.0:10,y in 1:1.0:10])
	if lin_2D(xs,xs,A,(1.0,1.0)) != 1.0 || lin_2D(xs,xs,A,(5.6,10.0)) != 56.0
		keep_working(md"`lin_2D` does not return a reasonable value at the edges.")
	elseif !isapprox(lin_2D(xs,xs,A,(5.4,5.4)),29.16) ||
		   !isapprox(lin_2D(xs,xs,A,(5.4,5.6)),30.24) 
		keep_working(md"`lin_2D` does not return the correct linear interpolation.")
	else
		correct()
	end
end

# ╔═╡ fa64d8cf-28f9-433f-9648-187920a7357e
let  x=Gray.(testimg)
	if Float64.(isinside((0,0),Gray.(testimg);linear=true))!=0.0 || 
	   Float64.(isinside((7,7),Gray.(testimg);linear=false))!=0.0
		keep_working(md"`isinside` does not return zero if x is outside of the img.")
	elseif !isapprox(Float64.(isinside((5.5,5.5),Gray.(testimg);linear=true)),
						0.3931372549019607) ||
		   !isapprox(Float64.(isinside((5.5,5.5),Gray.(testimg);linear=false)),
						0.49019607843137253)
		keep_working(md"`isinside` does not return the correct interpolated value for 
			a x inside of img.")
	else
		correct()
	end
end

# ╔═╡ 8081e714-0a48-4cc6-ab05-84c95f5a6889
let  x=[[0.607051  0.606921  0.574426  0.603138  0.596963  0.581312  0.587351  0.584621;
 0.598746  0.585663  0.543172  0.591666  0.612592  0.577408  0.603985  0.597484;
 0.6504    0.59256   0.596332  0.602961  0.625378  0.552507  0.556829  0.625526;
 0.600729  0.573677  0.618211  0.536227  0.629262  0.585435  0.588254  0.636256;
 0.597304  0.571668  0.586269  0.552671  0.613655  0.574334  0.555247  0.594499;
 0.60362   0.535435  0.589859  0.583221  0.582752  0.590071  0.56813   0.62699;
 0.556317  0.575208  0.612046  0.601466  0.597433  0.590125  0.549527  0.564352;
 0.548466  0.559405  0.546432  0.553916  0.555568  0.594887  0.54579   0.56213],
[ 0.4       0.403922  0.427451  0.439216  0.439216  0.439216  0.447059  0.443137;
 0.388235  0.411765  0.411765  0.427451  0.431373  0.431373  0.435294  0.435294;
 0.376471  0.396078  0.411765  0.415686  0.423529  0.427451  0.423529  0.423529;
 0.360784  0.380392  0.407843  0.415686  0.415686  0.419608  0.419608  0.419608;
 0.364706  0.376471  0.392157  0.407843  0.415686  0.411765  0.415686  0.419608;
 0.372549  0.376471  0.384314  0.403922  0.411765  0.411765  0.415686  0.423529;
 0.388235  0.384314  0.396078  0.403922  0.407843  0.415686  0.419608  0.419608;
 0.396078  0.403922  0.403922  0.403922  0.411765  0.419608  0.427451  0.431373],
[ 0.856049  0.85403   0.832609  0.841622  0.850682  0.873839  0.843663  0.832253;
 0.855654  0.833339  0.846113  0.841945  0.849131  0.86062   0.846804  0.830065;
 0.840134  0.855691  0.843374  0.843038  0.816519  0.836528  0.833075  0.852619;
 0.848861  0.834109  0.817608  0.845393  0.822977  0.828213  0.818379  0.846504;
 0.83869   0.822361  0.820966  0.833379  0.827212  0.832995  0.824186  0.828856;
 0.83215   0.833683  0.81945   0.813163  0.840935  0.832019  0.82272   0.815179;
 0.820078  0.811803  0.813684  0.819787  0.832453  0.824615  0.818011  0.822356;
 0.813522  0.812205  0.818636  0.824868  0.839347  0.82469   0.829257  0.825858]]
	if norm(Float64.(myrotation(shrekGray,pi/4;linear=true)[200:207,200:207])
			-x[1])/ norm(x[1])<0.1
		keep_working(md"`myrotation` seems to rotate in the wrong direction.")
	elseif norm(Float64.(myrotation(shrekGray,pi/4;linear=true)[200:207,200:207])
			-x[3])/	norm(x[3])>=0.05 ||
	       norm(Float64.(myrotation(shrekGray,pi/4;linear=false)[100:107,100:107])
			-x[2])/	norm(x[2])>=0.05
		keep_working(md"`myrotation` does not rotate the img exact enough.")
	else
		correct()
	end
end

# ╔═╡ f8ebec49-8df8-4496-a0c0-207a6b700148
let  x=[[ 0.596078  0.556863  0.541176  0.435294  0.388235  0.4       0.505882  0.572549;
 0.439216  0.486275  0.588235  0.592157  0.662745  0.639216  0.737255  0.752941;
 0.67451   0.72549   0.701961  0.780392  0.796078  0.85098   0.827451  0.847059;
 0.819608  0.831373  0.780392  0.819608  0.862745  0.839216  0.807843  0.807843;
 0.843137  0.807843  0.835294  0.780392  0.772549  0.772549  0.760784  0.74902;
 0.811765  0.776471  0.760784  0.764706  0.768627  0.776471  0.788235  0.788235;
 0.764706  0.784314  0.807843  0.811765  0.803922  0.788235  0.768627  0.796078;
 0.768627  0.784314  0.784314  0.796078  0.8       0.788235  0.807843  0.8],
[0.470588  0.470588  0.47451   0.486275  0.501961  0.509804  0.509804  0.509804;
 0.0       0.482353  0.482353  0.486275  0.498039  0.509804  0.521569  0.521569;
 0.0       0.0       0.494118  0.494118  0.498039  0.509804  0.521569  0.533333;
 0.0       0.0       0.0       0.505882  0.509804  0.513725  0.521569  0.533333;
 0.0       0.0       0.0       0.0       0.513725  0.513725  0.517647  0.52549;
 0.0       0.0       0.0       0.0       0.0       0.513725  0.513725  0.521569;
 0.0       0.0       0.0       0.0       0.0       0.0       0.517647  0.517647;
 0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.521569],
[0.623529  0.631373  0.607843  0.631373  0.682353  0.760784  0.65098   0.960784;
 0.686275  0.701961  0.568627  0.67451   0.537255  0.764706  0.701961  0.807843;
 0.737255  0.639216  0.639216  0.745098  0.698039  0.698039  0.745098  0.815686;
 0.67451   0.619608  0.615686  0.6       0.623529  0.709804  0.686275  0.862745;
 0.690196  0.647059  0.647059  0.607843  0.545098  0.631373  0.764706  0.862745;
 0.705882  0.717647  0.752941  0.580392  0.517647  0.635294  0.741176  0.811765;
 0.737255  0.701961  0.623529  0.690196  0.588235  0.678431  0.713725  0.592157;
 0.741176  0.686275  0.709804  0.501961  0.580392  0.588235  0.552941  0.870588]]
	if norm(Float64.(myshearing(shrekGray,0,1;linear=false))[140:147,280:287]
			-x[1])/ norm(x[1])<0.1
		keep_working(md"`myshearing` seems to shear in the wrong direction.")
	elseif norm(Float64.(myshearing(shrekGray,0,1;linear=true)[200:207,400:407])
			-x[3])/	norm(x[3])>=0.05 ||
	       norm(Float64.(myshearing(shrekGray,0,1;linear=false)[100:107,100:107])
			-x[2])/	norm(x[2])>=0.05
		keep_working(md"`myshearing` does not shear the img exact enough.")
	else
		correct()
	end
end

# ╔═╡ 19cbab80-0aad-47d0-8528-045537a1422e
let  x=[[ 0.267703  0.235937  0.209433  0.193377  0.183344  0.183616  0.233142;
 0.284098  0.257878  0.228649  0.198056  0.178584  0.187536  0.235208;
 0.288594  0.271835  0.247198  0.209755  0.180437  0.192965  0.230316;
 0.302879  0.27967   0.260399  0.226108  0.189026  0.17338   0.209574;
 0.318715  0.285725  0.262314  0.236247  0.199114  0.173768  0.184369;
 0.331761  0.295403  0.268829  0.245422  0.213041  0.175392  0.17031;
 0.343212  0.306493  0.277807  0.245115  0.223027  0.1853    0.165169],
[ 0.720929  0.732821  0.731977  0.731777  0.709357  0.716733  0.740537;
 0.72317   0.749399  0.742714  0.731168  0.730619  0.749965  0.710054;
 0.750581  0.743663  0.73475   0.722312  0.726443  0.723264  0.715113;
 0.743324  0.728222  0.737323  0.729431  0.701944  0.696804  0.705262;
 0.722214  0.689727  0.721978  0.720772  0.68028   0.677296  0.659021;
 0.747126  0.666754  0.697467  0.697099  0.669288  0.650715  0.629176;
 0.719393  0.683956  0.694774  0.689213  0.667739  0.660681  0.631013]]
	if norm(Float64.(prs3)[250:256,250:256]
			-x[2])/ norm(x[1])<0.1
		keep_working(md"`α,β,γ` seem to have the wrong sign.")
	elseif norm(Float64.(prs3)[250:256,250:256]
			-x[1])/	norm(x[1])>=0.05
		keep_working(md"`α,β,γ` are not quiet right.")
	else
		correct()
	end
end

# ╔═╡ Cell order:
# ╟─3e968320-e085-4e67-9111-f1d4b3292745
# ╠═649dbe1a-2da5-11ec-3f4e-39b5d6dfe737
# ╟─d5e0b830-34fc-4666-8e71-71cef73f2cba
# ╟─7d88ae11-db3c-4b4e-819d-dddfccd61a46
# ╟─27d1f36f-6329-4030-b7a9-f8ee0630ffe8
# ╟─c6de2614-04b5-4030-a814-835dbf5bc02d
# ╟─7ad27580-fb5b-4908-9944-3b89607571b9
# ╠═6c23560b-010f-4034-88e9-11625514bf15
# ╟─1a087ab6-46d2-4835-9ae3-b9c259ed84dd
# ╟─e3d8cf3f-abe6-4e56-8c90-d6ed87188376
# ╠═bf86aa2d-ff6e-4063-b9eb-bb43813c2380
# ╟─5898be63-30fd-42d6-8773-49bf53e48e07
# ╟─516cc797-aa1c-4c53-a672-2c7c41077434
# ╠═2555fea0-162c-4d36-8a50-80f609aa0399
# ╟─fe20e217-da71-472d-9334-bfebee8ce07a
# ╟─ea5498b1-90b8-47d0-8e8a-d40e77c0dbab
# ╠═f100b3b5-45c1-44b2-8f86-9da0622a5ebd
# ╟─1b7793cd-c2f4-4420-ba5b-e1286d667cba
# ╟─fff1ecce-1f11-4b56-b0ba-be895891c8e5
# ╟─306971e0-d544-4e17-86f8-021151c30445
# ╠═280315a1-5831-4e6f-b3bd-387e85e8f8fe
# ╟─3b4ae844-efa7-452a-aacb-c3a0b6fccc6e
# ╟─b95bbc90-24c7-4c57-906f-b82efad06ca1
# ╟─aed1bc0a-0bf6-4fda-8959-2cb021d1a8ad
# ╠═9f6dcc3d-b3dc-400d-8d69-44e82cf7b6e1
# ╟─5c89a853-59ec-484c-bd3f-bb3f5932bb08
# ╟─fda0e2ca-664c-4a66-96a5-38f5c7d22499
# ╟─82dca013-560c-4bee-a894-71b6ed4ca5c7
# ╟─b80ae2d8-5574-49dc-92cb-063df31d2b89
# ╠═af2cf3fd-bdf0-45b3-8d87-4f828f87fd1f
# ╟─74a7342e-24ac-4a3c-81a5-73dab7f878b2
# ╟─de91fefd-229e-45a1-8573-11570228d428
# ╠═51587457-5b8a-4c5a-89a9-dbf9fd9ec5ef
# ╟─65b29352-d6f4-4008-bc26-85c2abd293e6
# ╟─1f1d5598-a26f-4875-8642-ec20750dd252
# ╠═37dde329-0ef5-4e77-a6b4-e62b0ce425ff
# ╟─152c07d3-6c14-4ac4-bf16-1fb26b5b1dbd
# ╟─078dc45c-97ea-408d-af4b-954f41edf53f
# ╠═3b59e67e-4f24-4eeb-84a1-a22c5d4a32ad
# ╟─4106e8e6-b414-4bf6-8f1c-89bd3af44a1d
# ╟─b27a908f-eb2d-47ed-b2eb-f9b7418b82c1
# ╟─9bf6df9d-bdeb-461d-b4f4-669e29136470
# ╠═645cc476-85df-4bb3-b593-b367911f98cb
# ╟─fa64d8cf-28f9-433f-9648-187920a7357e
# ╟─98d5a3a6-233e-4902-89f3-c0592c9e828d
# ╠═6bae3241-79d4-475d-8989-c8bfd541debc
# ╟─8081e714-0a48-4cc6-ab05-84c95f5a6889
# ╟─50d41a88-ed7d-486a-8af9-7f93eac51ffd
# ╟─f93d09f8-f10d-4d13-baa4-9593dcf31531
# ╠═f55360b2-03bd-41d1-85d3-353bef0b4d52
# ╟─f8ebec49-8df8-4496-a0c0-207a6b700148
# ╟─993059bf-0da5-45f3-a59a-ed40d72c993d
# ╟─cc609e7a-d56d-4977-a433-d9a30ecbf4d2
# ╟─2c0613ee-bfb6-476e-97a3-7cab7df902df
# ╟─0fd17c6a-5eb3-4b19-88f1-a509265c0e4d
# ╟─19cbab80-0aad-47d0-8528-045537a1422e
# ╟─448cb1b3-8fc0-425e-95aa-7eaa4c4a9acd

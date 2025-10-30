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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CoordinateTransformations = "150eb455-5306-5404-9cee-2592286d6298"
ImageTransformations = "02fcd773-0e25-5acc-982a-7f6622650795"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Rotations = "6038ab10-8711-5258-84ad-4b1120ba62dc"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
TestImages = "5e47fb64-e119-507b-a336-dd2b206d9990"

[compat]
CairoMakie = "~0.12.5"
CoordinateTransformations = "~0.6.3"
ImageTransformations = "~0.9.5"
Images = "~0.25.3"
Interpolations = "~0.14.7"
PlutoUI = "~0.7.59"
Rotations = "~1.7.1"
StaticArrays = "~1.9.7"
TestImages = "~1.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.10"
manifest_format = "2.0"
project_hash = "e5a9bb3c6a36828ae01dcaafad102ee34d46fa25"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "361dec06290d76b6d70d0c7dc888038eec9df63a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "eaa040768ea663ca695d442be1bc97edfe6824f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.3+0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "8e233d5167e63d708d41f87597433f59a0f213fe"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.4"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "294e99f19869d0b0cb71aef92f19d03649d028d5"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "3447781d4c80dbe6d71d239f7cfb1f8049d4c84f"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.6"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "8e64ab2f0da7b928c8ae889c514a52741debc1c2"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.2"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Bzip2_jll", "FFTW_jll", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "Zstd_jll", "libpng_jll", "libwebp_jll", "libzip_jll"]
git-tree-sha1 = "d670e8e3adf0332f57054955422e85a4aec6d0b0"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.2005+0"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "44664eea5408828c03e5addb84fa4f916132fc26"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.1"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "5fa9f92e1e2918d9d1243b1131abe623cdf98be7"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.3"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "bf0210c01fb7d67c31fed97d7c1d1716b98ea689"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.1"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "89e1e5c3d43078d42eed2306cab2a11b13e5c6ae"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.54"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "8e6a74641caf3b84800f2ccd55dc7ab83893c10b"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.17.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "204f06860af9008fa08b3a4842f48116e1209a2c"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.9"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "b0e2e3473af351011e598f9219afb521121edd2b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.6"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "a370fef694c109e1950836176ed0d5eabbb65479"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "567515ca155d0020a45b05175449b499c63e7015"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "7dc7028a10d1408e9103c0a77da19fdedce4de6c"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f19301ae653233bc88b1810ae908194f07f8db9d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

    [deps.Rotations.weakdeps]
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "3e5f165e58b18204aed03158664c4982d691f454"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.5.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringDistances]]
deps = ["Distances", "StatsAPI"]
git-tree-sha1 = "5b2ca70b099f91e54d98064d5caf5cc9b541ad06"
uuid = "88034a9c-02f8-509d-84a9-84ec65e18404"
version = "0.11.3"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TestImages]]
deps = ["AxisArrays", "ColorTypes", "FileIO", "ImageIO", "ImageMagick", "OffsetArrays", "Pkg", "StringDistances"]
git-tree-sha1 = "0567860ec35a94c087bd98f35de1dddf482d7c67"
uuid = "5e47fb64-e119-507b-a336-dd2b206d9990"
version = "1.8.0"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "38f139cc4abf345dd4f22286ec000728d5e8e097"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "cec2df8cf14e0844a8c4d770d12347fda5931d72"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.25.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

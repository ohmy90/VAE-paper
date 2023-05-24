# VAE-paper

Generating numbers with variational autoencoders






Brian Duong






Probability and statistics, 201-HTH-05 section 00001
Differential equations, 201-HTL-VA section 00001








May 26, 2023























Table of content

1	Abstract (missing)
	
2	Overview of neural networks and autoencoders
	2.1 Neural networks
	2.2 Autoencoders

3	Variational autoencoders (VAE)
	3.1 Functions of a VAE
	3.2 Defining VAEs mathematically
		3.2.1 Encoder and Evidence Lower Bound (ELBO)
		3.2.2 Two for one
		3.2.3 Stochastic Gradient-Based optimisation of the ELBO
		3.2.4 Reparameterization trick
		3.2.5 Models of the encoder and the decoder
		3.2.6 Mean and variance nodes
	3.3 Issues with standard VAEs ?
	3.4 Possible fixes ?

4	Using variational autoencoders to generate numbers (missing)

5	Conclusion (missing)

6	Bibliography (missing)




































2. Overview of neural networks and autoencoders

2.1 Neural networks
To gain a better understanding of the subject, it is essential to comprehend a vital component of machine learning: neural networks. A neural network is a system of artificial neurons or nodes that is structured in a way similar to the human brain. A deep neural network would consist of an input layer, multiple hidden layers, and an output layer. Each node is connected to another node of the next layer and has a correlated weight and threshold that can be altered to obtain different results in the output layer. When a node gets activated, it sends data to the next layer of the network only if its output surpasses the threshold value. Neural networks require training data to learn and improve. Once adjusted, they enable us to rapidly classify, cluster data and solve artificial intelligence problems.

Figure 2.1: Representation of a deep neural network

2.1 Autoencoders
	Autoencoders are a type of neural network where we introduce a bottleneck within the hidden layers which forces a compressed representation of the initial dataset that contains essential knowledge.

Figure 2.2: Representation of a simple autoencoder

The aim of an autoencoder is to reconstruct the input data from this compressed portrayal as accurately as possible . It consists of three main components: an encoder, a decoder and a latent space. The encoder seeks to compress the high dimensional dataset to a lower dimension, which is then stored in the hidden space. By doing so, the neural network is “forced” to learn a latent representation of the input data, which then, the decoder reconstructs the data using that latent knowledge. However, autoencoders have difficulties in terms of generating new and diverse data. This is because the encoder takes a deterministic approach during compression. This means that the encoded data in the latent space may not be continuous or allow straightforward interpolation.

Figure 2.3: Representation of a latent space of an autoencoder trained on the MNIST dataset. The autoencoder has difficulties generating a new picture between the 1 and 7 data sets since there are large “gaps” between the two.

3. Variational autoencoders (VAEs)

3.1 Function of a VAE
	In order to generate unique and varied data, we make use of variational autoencoders (VAEs). What differentiates them from ordinary autoencoders is that VAEs take a probabilistic approach to representing latent data. In other words, the probabilistic encoder maps the input data to a probability distribution in the lower-dimensional space and the probabilistic decoder maps the sampled latent variables back to the original space. As such, their latent spaces are continuous, which enables easy random sampling and interpolation. 

3.2 Defining VAEs mathematically
To enable this continuity, we implement mean and variance nodes within the encoder, which will be addressed later in detail.

Figure 3.1: Illustration of a VAE model with the multivariate Gaussian assumption

Let  and  be the parameters of the encoder and the decoder, respectively, and z the latent variables. We define x to be the vector of all observed variables and x’ the reconstructed vector. We want x’ to be approximately x, and thus p(x)p*(x), where p*(x) is the joint distribution we want to model and p(x) is a model with parameter .

We can define the relationship between x and z to be a joint distribution p(x,z) and by: 	Prior p(z) 				Likelihood  p(xlz)			Posterior  p(zlx)
The marginal distribution over the observed variables is called the marginal likelihood and is given by:
p(x) = p(x,z) dz 
We can define p(x,z) to be equal to p(z)(p(xlz)) where the prior distribution and/or the likelihood distribution are specified. The main issue is that the marginal likelihood is intractable, making the posterior distribution p(zlx) =p(x,z)p(x) also intractable.
3.2.1 Encoder and Evidence Lower Bound (ELBO)
	To be able to compute the posterior distribution p(zlx), we introduce an encoder, or also called an inference model, q(zlx). We want q(zlx)p(zlx). Additionally, the encoder q(zlx) can be parametrized using deep neural networks. This will help the encoder to capture the complex patterns and features in the input data. 
	We want to give a lower bound to the marginal likelihood p(x) in order to mathematically optimize p(x) which would be called the evidence lower bound, or ELBO for short. As such, the marginal likelihood can be lower-bounded by the ELBO, which is given by:
	
The second term in the final equation is called the Kulllback-Leibler (KL) divergence between q(zlx) and p(zlx), and is non-negative. The first term is the ELBO, which can be defined as such:			L,(x) = Eq(zlx) [log p(x,z)-log q(zlx)]. 
Because the KL divergence is non-negative, the ELBO is a lower bound on the log marginal likelihood. Meaning, L,(x) log p(x). Fundamentally, the KL divergence determines two “distances”:
The distance between the approximate posterior and the true posterior
The gap between the ELBO and the marginal likelihood  log p(x). Therefore, the better the approximation between the approximate posterior and the true posterior, the smaller the gap.
3.2.2 Two for one
	By optimizing the ELBO with respect to parameters  and , we optimize two things:
It will maximize the marginal likelihood p(x), which means a better decoder
It will minimize the KL divergence of the approximation between the two posteriors, meaning a better encoder q(zlx).
3.2.3 Stochastic Gradient-Based optimisation of the ELBO
	If we take the unbiased gradients of the ELBO with respect to , we obtain the following: 	∇L,(x) = ∇Eq(zlx) [log p(x,z)-log q(zlx)]
   = Eq(zlx) [∇(log p(x,z)-log q(zlx))]
    ∇(log p(x,z)-log q(zlx))
   = ∇log p(x,z)
	The last equation is a Monte Carlo estimator, which means we take z to be a random sample from the encoder q(zlx). However if we try to take the gradient with respect to , we encounter a problem due to q(zlx) being a function of , meaning we cannot move the gradient past the expectation: ∇L,(x) = ∇Eq(zlx) [log p(x,z)-log q(zlx)]
     Eq(zlx) [∇(log p(x,z)-log q(zlx))]
3.2.4 Reparameterization trick
	In order to find the gradient, we make use of a trick where we make a change of variables. Given z and , we express z q(zlx) as a smooth differentiable transformation of another random variable : z = g(, , x) where  is independent of  or x. By doing so, we are able to “express” the randomness within z as  which will allow us to compute the gradient.

Figure 3.2: Illustration of the reparameterization trick
The ELBO can then be rewritten as:
 L,(x) = Ep() [log p(x,z)-log q(zlx)] where z = g(, , x)
We can form a simple Monte Carlo estimator L',(x) for the ELBO of each data point. This estimator is based on a sample of noise  drawn from the distribution p().
z = g(, , x)
L',(x) = [log p(x,z)-log q(zlx)]
Additionally, the gradient is an unbiased estimator of the exact ELBO for a single data point. As such, when averaged over multiple noise samples, the estimated gradient is equal to the true gradient:
 Ep() [∇,L',(x;)] = Ep()[∇,(log p(x,z)-log q(zlx))]
		         =∇,L,(x) 
3.2.5 Models of q(zlx) and  p(x,z)
	There are multiple different models that can be used for the encoder and the decoder , each suited for specific goals. As long as the right transformation g() is chosen, log q(zlx) is relatively easy to compute. The chosen models used to generate numbers in this paper are a standard Gaussian encoder and a factorized Bernoulli distribution is used as the decoder.

3.2.6 Mean and variance nodes
	Since the encoder q(zlx)  creates a probabilistic distribution in the latent space and we are using a standard Gaussian encoder to generate unique and diverse pictures, we introduce mean and variance nodes within the encoder. In other words, the encoder outputs two vectors: μ and σ. The mean vector determines the position for the input encoding, and the standard deviation determines how much from the mean the encoding can deviate. As such, the decoder learns that all the points within the “area” of the distribution are all equally the same class. This allows the latent space to be continuous and fill in the “gaps”. 






Figure 3.3: Illustration of the mean vector and the variance vector within a latent space
4. Generating numbers with variational autoencoders

4.1 Code
	The code used is a basic VAE example from the Auto-Encoding Variational Bayes paper from Kingma and Welling. The models used are mentioned in 3.2.5. 

4.2 Process
4.3 Results


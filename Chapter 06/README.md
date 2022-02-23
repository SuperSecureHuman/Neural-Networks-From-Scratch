# Introducing Optimization

Now that we have a built a network, that can perfrom one forward propogate, and calculate the loss, its time to introduce a method that will help us to change the weights and biases of the network. This is the hardest part of a neural netowork.

First, we can think to use multiple random values, and stick with the one that gives us the least error. Lets actually try making something like that.

Rest of the chapter will be continued in the notebook.

Basically, we are just incrementing the weights and biases by a small amount, and then saving the version with the best accuracy.

[Optimization With Random Values](./1.Random_Optimization.ipynb)

---

End of Chapter 06 
## 💡 Notes on Computational Cost and Training Time

### 1. Per-Step Cost Across Methods

All three methods (DRQI, IPMNN, DRM) follow a similar structure during each training step:  
- one forward pass through the neural network,  
- some arithmetic operations,  
- and one backward pass for gradient computation.

In practical applications, especially with larger networks used to improve approximation accuracy, **most of the computation time is spent inside the neural network itself**. The arithmetic calculations outside the network (like residual or loss evaluations) are relatively lightweight.

So, in most situations, it's fair to assume that **each training step takes about the same time** across different methods.

That said, in a few of our experiments (notably in the QMC results), we did observe some unusually high average training times per epoch.  
This was traced back to hardware conditions — specifically, the machine was running in **battery-saving mode without being plugged in**, which significantly slowed down CPU and GPU performance.😂

---

### 2. Epochs vs. Training Time

We also ran some additional tests to check whether plotting results against **actual training time** (instead of the number of epochs) would change any conclusions.  

The answer: **Not really.**  
Because the per-step runtime is nearly the same for all methods, the trends stay almost identical whether you use "epochs" or "time" on the x-axis.

So, to keep the main paper cleaner, we used "epoch number" throughout.  
Just for reference:  
In the 1D Laplace case, each epoch typically takes about **0.007–0.009 seconds**, though this can fluctuate depending on hardware status and system load.

---

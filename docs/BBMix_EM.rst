Derivations of EM steps for Mixture of Beta Binomial Distribution
-----------------------------------------------------------------

:author: Chen Qiao

-  Beta-Binomial Distribution (BB):

   .. math::


      \begin{align}
      p(y^{(i)} | \theta) &= \int_{0}^{(1)}Bin(y^i|n,p) \cdot Beta(p|\alpha, \beta) dp \\
                    &= {{n}\choose{y^{(i)}}}\frac{1}{B(\alpha,\beta)} \int_{0}^{1} p^{y^{(i)}+ \alpha -1} (1-p)^{n - y^{(i)} + \beta - 1} dp   \\
                    &= {{n}\choose{y^{(i)}}}\frac{B(y^{(i)}+\alpha, n-y^{(i)}+\beta)}{B(\alpha, \beta)}
      \end{align}

-  Mixture of Beta-Binomial Distribution (MBB), joint probablility,
   suppose there are K components, and let the variables be
   :math:`\mathbf{\gamma} = [\gamma_{0}, \gamma_{1}, ..., \gamma_{k}, \gamma_{K}]`,
   :math:`\gamma_{k} \text{~} Bin(\gamma_{k}|\pi_k)`:

   .. math::


      \begin{align}
      p(y^{(i)}, \gamma^{(i)} | \theta, \pi) &= \prod_{k=1}^{K} p(y^{(i)}, \gamma_{k}^{(i)}| \theta_{k}, \pi_k)^{\gamma^{(i)}_k} \\
                                                          &= \prod_{k=1}^{K} \{p(y^{(i)} | \theta_{k}) \pi_k \}^{\gamma_k^{(i)}}
      \end{align}

-  Log likelihood of the full data (MBB):

   .. math::


      \begin{align}
      \log p(y^{(i)}, \gamma^{(i)} | \theta, \pi) &= \sum_{k=1}^{K} \bigg \{ \gamma^{(i)}_k \big(\log\pi_k + \log p(y^{(i)} | \theta_{k}) \big)
      \bigg\} 
      \end{align}

E step:
-------

.. math::


   \begin{align}
   E_{\gamma^{(i)}_k \text{~} p(\gamma_k|y^{(i)})}[\log p(y^{(i)}, \gamma^{(i)}] &= E\bigg[\sum_{k=1}^{K} \big \{ \gamma^{(i)}_k \big(\log\pi_k + \log p(y^{(i)} | \theta_k) \big)\big\} \bigg] \\
                         &= \sum_{k=1}^{K} \big \{ E[\gamma^{(i)}_k] \big(\log\pi_k + \log p(y^{(i)} | \theta_k) \big)\big\} \\
   \end{align}

As:

.. math::


   \begin{align}
   \bar{\gamma}_k^{(i)} = E(\gamma^{(i)}_k|y^{(i)}, \theta, \pi) &= p(\gamma_k = 1| y^{(i)}, \theta, \pi)   \text(- Expection of Bernoulli distribution)\\
                                          &= \frac{p(\gamma^{(i)}_k=1, y^{(i)}| \theta, \pi)}{\sum_{k=1}^{K}p(\gamma_k^{(i)}=1, y^{(i)}| \theta, \pi)} \\
                                          &= \frac{p(y^{(i)}|\gamma^{(i)}_k=1, \theta, \pi) \cdot p(\gamma^{(i)}_k=1|\pi)}{\sum_{k=1}^{K}p(y^{(i)}|\gamma^{(i)}_k=1, \theta, \pi) \cdot p(\gamma^{(i)}_k=1|\pi)} \\
                                          &= \frac{ {{n}\choose{y^{(i)}}} \frac{B(y^{(i)} + \alpha_k, n-y^{(i)} + \beta_k)}{B(\alpha_k, \beta_k)} \cdot \pi_k}{ \sum_{k=1}^{K}{{n}\choose{y^{(i)}}} \frac{B(y^{(i)} + \alpha_k, n-y^{(i)} + \beta_k)}{B(\alpha_k, \beta_k)} \cdot \pi_k} \\
                                          &= \frac{\frac{B(y^{(i)} + \alpha_k, n-y^{(i)} + \beta_k)}{B(\alpha_k, \beta_k)} \cdot \pi_k}{ \sum_{k=1}^{K} \frac{B(y^{(i)} + \alpha_k, n-y^{(i)} + \beta_k)}{B(\alpha_k, \beta_k)} \cdot \pi_k}
   \end{align}

M step
------

M-step hence is to optimize:

.. math::


    \max_{\pi, \theta}\bigg\{\sum_{k=1}^{K} \big \{ \bar{\gamma}_k^{(i)} \big(\log\pi_k + \log p(y^{(i)} | \theta_k) \big)\big\} \bigg\} \\
    = \max_{\pi, \theta}\bigg\{ \sum_{k=1}^{K} \big \{ \bar{\gamma}_k^{(i)} \big(\log\pi_k + \log{{n}\choose{y^{(i)}}} + \log B(y^{(i)}+\alpha_k, n-y^{(i)}+\beta_k) - \log{B(\alpha_k, \beta_k)} \big)\big\} \bigg\} \\
   s.t. \sum_{k=1}^{K} \pi_k = 1, \\ \pi_k \geq 0, k=1,...,K\\ \alpha > 0, \\ \beta > 0

Using all the data points:

.. math::


   \max_{\pi, \theta}\bigg\{ \sum_{i=1}^{N}\sum_{k=1}^{K} \big \{ \bar{\gamma}_k^{(i)} \big(\log\pi_k + \log{{n}\choose{y^{(i)}}} + \log B(y^{(i)}+\alpha_k, n-y^{(i)}+\beta_k) - \log{B(\alpha_k, \beta_k)} \big)\big\} \bigg\} \\
   s.t. \sum_{k=1}^{K} \pi_k = 1 \\ \pi_k \geq 0, k=1,...,K \\ \alpha > 0, \\ \beta > 0


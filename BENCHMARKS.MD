### Benchmarks ran with the latest .exe file at the time, benchmark.py script, no encoding and on a 1920x1080p video with FP16 or BF16 on for all runs.

#### Note, since these are ran in order with no time gaps inbetween there's a possibility for GPU heat becoming a bottleneck so these results may not be fully representative of the real world performance.

#### If the result is 0.0 that's typically because of Out Of VRAM Issues ( like GMFSS )
-------------------------------------------------------------------------------------------------------------

# 13700k Undervolted + Overclocked, 3090 Overclocked, 32GB DDR5 6800Mhz, TAS 1.5.0, Nilas.
<div style="display: flex; justify-content: space-between;">
<div style="flex: 1;">




-------------------------------------------------------------------------------------------------------------

# AMD 3700x, RTX 3080, 24GB DDR4 3600MHz, TAS 1.5.0, Windows VM ( 7C/14Threads ), TNTWise.

<div style="display: flex; justify-content: space-between;">
<div style="flex: 1;">


## Denoise

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>scunet</td>
      <td>1.4</td>
    </tr>
    <tr>
      <td>nafnet</td>
      <td>2.7</td>
    </tr>
    <tr>
      <td>dpir</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>span</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>

</div>
</div>

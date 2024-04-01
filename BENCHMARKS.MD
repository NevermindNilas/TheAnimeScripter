### Benchmarks ran with the latest .exe file at the time, benchmark.py script, no encoding and on a 1920x1080p video with FP16 or BF16 on for all runs.

#### Note, since these are ran in order with no time gaps inbetween there's a possibility for GPU heat becoming a bottleneck so these results may not be fully representative of the real world performance.

#### If the result is 0.0 that's typically because of Out Of VRAM Issues ( like GMFSS )
-------------------------------------------------------------------------------------------------------------

# 13700k Undervolted + Overclocked, 3090 Overclocked, 32GB DDR5 6800Mhz, TAS 1.5.0, Nilas.
<div style="display: flex; justify-content: space-between;">
<div style="flex: 1;">

## Upscale

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>shufflecugan</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>cugan</td>
      <td>6.3</td>
    </tr>
    <tr>
      <td>compact</td>
      <td>11.0</td>
    </tr>
    <tr>
      <td>ultracompact</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>superultracompact</td>
      <td>36.0</td>
    </tr>
    <tr>
      <td>span</td>
      <td>9.5</td>
    </tr>
    <tr>
      <td>span-ncnn</td>
      <td>5.7</td>
    </tr>
    <tr>
      <td>cugan-ncnn</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>omnisr</td>
      <td>1.1</td>
    </tr>
    <tr>
      <td>realesrgan</td>
      <td>4.3</td>
    </tr>
    <tr>
      <td>realesrgan-ncnn</td>
      <td>5.9</td>
    </tr>
    <tr>
      <td>shufflecugan-ncnn</td>
      <td>1.3</td>
    </tr>
    <tr>
      <td>apisr</td>
      <td>4.4</td>
    </tr>
  </tbody>
</table>

</div>
<div style="flex: 1;">

## Interpolate

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>rife4.6</td>
      <td>122.0</td>
    </tr>
    <tr>
      <td>rife4.6-ensemble</td>
      <td>71.0</td>
    </tr>
    <tr>
      <td>rife4.14</td>
      <td>88.0</td>
    </tr>
    <tr>
      <td>rife4.14-ensemble</td>
      <td>56.0</td>
    </tr>
    <tr>
      <td>rife4.15</td>
      <td>88.0</td>
    </tr>
    <tr>
      <td>rife4.15-ensemble</td>
      <td>56.0</td>
    </tr>
    <tr>
      <td>rife4.16-lite</td>
      <td>111.0</td>
    </tr>
    <tr>
      <td>rife4.16-lite-ensemble</td>
      <td>70.0</td>
    </tr>
    <tr>
      <td>rife4.6-ncnn</td>
      <td>75.0</td>
    </tr>
    <tr>
      <td>rife4.6-ncnn-ensemble</td>
      <td>44.0</td>
    </tr>
    <tr>
      <td>rife4.14-ncnn</td>
      <td>55.0</td>
    </tr>
    <tr>
      <td>rife4.14-ncnn-ensemble</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>rife4.15-ncnn</td>
      <td>55.0</td>
    </tr>
    <tr>
      <td>rife4.15-ncnn-ensemble</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>gmfss</td>
      <td>7.8</td>
    </tr>
    <tr>
      <td>gmfss-ensemble</td>
      <td>7.8</td>
    </tr>
  </tbody>
</table>

</div>
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
      <td>1.8</td>
    </tr>
    <tr>
      <td>nafnet</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>dpir</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>span</td>
      <td>9.6</td>
    </tr>
  </tbody>
</table>

</div>
</div>


-------------------------------------------------------------------------------------------------------------

# AMD 3700x, RTX 3080, 24GB DDR4 3600MHz, TAS 1.5.0, Windows VM ( 7C/14Threads ), TNTWise.

<div style="display: flex; justify-content: space-between;">
<div style="flex: 1;">

## Upscale

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>shufflecugan</td>
      <td>13.0</td>
    </tr>
    <tr>
      <td>cugan</td>
      <td>4.7</td>
    </tr>
    <tr>
      <td>compact</td>
      <td>8.8</td>
    </tr>
    <tr>
      <td>ultracompact</td>
      <td>15.0</td>
    </tr>
    <tr>
      <td>superultracompact</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>span</td>
      <td>7.5</td>
    </tr>
    <tr>
      <td>span-ncnn</td>
      <td>3.8</td>
    </tr>
    <tr>
      <td>cugan-ncnn</td>
      <td>0.3</td>
    </tr>
    <tr>
      <td>omnisr</td>
      <td>0.9</td>
    </tr>
    <tr>
      <td>realesrgan</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>realesrgan-ncnn</td>
      <td>4.1</td>
    </tr>
    <tr>
      <td>shufflecugan-ncnn</td>
      <td>0.6</td>
    </tr>
    <tr>
      <td>apisr</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>

</div>
<div style="flex: 1;">

## Interpolate

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>rife4.6</td>
      <td>87.0</td>
    </tr>
    <tr>
      <td>rife4.6-ensemble</td>
      <td>51.0</td>
    </tr>
    <tr>
      <td>rife4.14</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>rife4.14-ensemble</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>rife4.15</td>
      <td>66.0</td>
    </tr>
    <tr>
      <td>rife4.15-ensemble</td>
      <td>43.0</td>
    </tr>
    <tr>
      <td>rife4.16-lite</td>
      <td>76.0</td>
    </tr>
    <tr>
      <td>rife4.16-lite-ensemble</td>
      <td>51.0</td>
    </tr>
    <tr>
      <td>rife4.6-ncnn</td>
      <td>44.0</td>
    </tr>
    <tr>
      <td>rife4.6-ncnn-ensemble</td>
      <td>29.0</td>
    </tr>
    <tr>
      <td>rife4.14-ncnn</td>
      <td>36.0</td>
    </tr>
    <tr>
      <td>rife4.14-ncnn-ensemble</td>
      <td>25.0</td>
    </tr>
    <tr>
      <td>rife4.15-ncnn</td>
      <td>35.0</td>
    </tr>
    <tr>
      <td>rife4.15-ncnn-ensemble</td>
      <td>24.0</td>
    </tr>
    <tr>
      <td>gmfss</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>gmfss-ensemble</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>

</div>
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

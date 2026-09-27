BiRefNet inference architecture is adapted from
[ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) at commit
`ebcc0bc8ec7fe919cec829f2dea656b3078acddc` (MIT license in `LICENSE`).

The package keeps the Swin Tiny backbone, BiRefNet decoder, and modules needed
to load the `Zarxrax/BiRefNet-Real_Anime` Lite checkpoint. Imports were made
package-relative, the configuration was fixed to that checkpoint, and training
and Hugging Face Hub integration were removed. The checkpoint itself is
published by Zarxrax under Apache-2.0 and is downloaded separately.

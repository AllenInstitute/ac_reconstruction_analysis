# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:06:00 2023

@author: kevint
"""
import argschema


class FilterSkeletonsOptions(argschema.schemas.DefaultSchema):
    names = argschema.fields.List(
        argschema.fields.Str, required=False, allow_none=True)
    ids = argschema.fields.List(
        argschema.fields.Str, required=False, allow_none=True)
    mincablelength = 
    deskew_flip = argschema.fields.Bool(required=False, default=True)
    deskew_crop = argschema.fields.Float(required=False, default=1.0)


class TiffDirToZarrInputParameters(argschema.ArgSchema,
                                   TiffDirToNGFFParameters):
    chunk_size = argschema.fields.Tuple((
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int(),
        argschema.fields.Int()), required=False, default=(1, 1, 64, 64, 64))



class TiffDirToZarr(argschema.ArgSchemaParser):
    default_schema = TiffDirToZarrInputParameters

    def run(self):
        deskew_options = (self.args["deskew_options"]
                          if "deskew_options" in self.args else {})
        tiffdir_to_ngff_group(
            self.args["input_dir"], self.args["output_format"],
            self.args["output_file"], self.args["group_names"],
            self.args["group_attributes"],
            self.args["max_mip"],
            self.args["mip_dsfactor"],
            self.args["chunk_size"],
            concurrency=self.args["concurrency"],
            compression=self.args["compression"],
            lvl_to_mip_kwargs=self.args["lvl_to_mip_kwargs"],
            deskew_options=deskew_options)


class TiffDirToN5(TiffDirToZarr):
    default_schema = TiffDirToN5LegacyParameters


if __name__ == "__main__":
    mod = TiffDirToZarr()
    mod.run()
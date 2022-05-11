from ..sources import create_image as create_source_image


class SceneImageMixin(object):
    def _create_image(self, da_scene, da_src):
        data_source = self.data_source

        if self.aux_name is None:
            source_name = data_source.source
            product = data_source.type
            product_name = data_source.name
        else:
            source_name = self.data_source.aux_products[self.aux_name]["source"]
            product = self.data_source.aux_products[self.aux_name]["type"]
            product_name = self.aux_name

        if source_name == "goes16" and product == "truecolor_rgb":
            # to be able to create a RGB image with satpy we need to set the
            # attrs again to ensure we get a proper RGB image
            da_scene.attrs.update(da_src.attrs)

        # if self.aux_name is not None:
        # invert_colors = data_source.aux_products[self.aux_name].get("invert_values_for_rgb", False)
        img_domain = create_source_image(
            da_scene=da_scene,
            source_name=source_name,
            product=product,
            context=dict(datasource_path=self.data_path, product_name=product_name),
        )

        return img_domain

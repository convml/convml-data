from ..sources import create_image as create_source_image
from ..sources import user_functions


class SceneImageMixin(object):  # noqa
    @property
    def image_function(self):
        if self.aux_name is not None:
            image_function = self.data_source.aux_products[self.aux_name].get(
                "image_function", "default"
            )
        else:
            image_function = self.data_source._meta.get("image_function", "default")
        return image_function

    def _create_image(self, da_scene):
        data_source = self.data_source

        if self.aux_name is None:
            source_name = data_source.source
            product_name = data_source.product
        else:
            source_name = self.data_source.aux_products[self.aux_name]["source"]
            product_name = self.data_source.aux_products[self.aux_name]["product"]

        image_function = self.image_function

        if image_function is None:
            raise Exception("Shouldn't call ._create_image() if image_function == None")

        if image_function != "default":
            image_function = user_functions.get_user_function(
                context=dict(
                    datasource_path=self.data_path, product_identifier=image_function
                )
            )

        img_domain = create_source_image(
            da_scene=da_scene.squeeze(),
            source_name=source_name,
            product=product_name,
            image_function=image_function,
        )

        return img_domain

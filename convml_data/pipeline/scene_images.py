from ..sources import create_image as create_source_image
from ..sources import user_functions


class SceneImageMixin(object):  # noqa
    @property
    def image_function(self):
        if self.aux_name is not None:
            if "__extra__" in self.aux_name:
                image_function = None
            else:
                image_function = self.data_source.aux_products[self.aux_name].get(
                    "image_function", None
                )
        else:
            image_function = self.data_source._meta.get("image_function", "default")
        return image_function

    def _create_image(self, da_scene):
        image_function = self.image_function

        if image_function is None:
            raise Exception("Shouldn't call ._create_image() if image_function == None")

        if image_function != "default":
            image_function = user_functions.get_user_function(
                context=dict(
                    datasource_path=self.data_path, product_identifier=image_function
                )
            )

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            img_domain = create_source_image(
                da_scene=da_scene.squeeze(),
                source_name=self.source_name,
                product=self.product_name,
                image_function=image_function,
            )

        return img_domain

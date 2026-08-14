//! Provider logo cell backed by packaged `asset://login/<provider>` PNGs,
//! degrading to a themed monogram card when no logo is packaged.

use omp_core::str::IntoStr;

use crate::{
	Component, Dim, Graphics, PaintCtx, Prop, Props, Rect, Slot, UiContext,
	components::{Col, Img},
	imagereg,
};

/// A provider logo cell using packaged assets.
///
/// `id` selects the provider; `w`/`h` bound the cell box (defaults 4×2).
/// Pixel-capable terminals render the interned PNG; cell terminals sample it
/// into half-block cells; providers without a packaged logo render a bold
/// monogram derived from the id.
pub struct Logo {
	props: Props,
	slot:  Slot,
	inner: LogoInner,
}

/// Resolved backing content; concrete variants keep layout and paint
/// dispatch allocation-free.
enum LogoInner {
	/// Not yet resolved against the asset table.
	Pending,
	/// Packaged PNG rendered through [`Img`].
	Image(Img),
	/// Monogram fallback for providers without a packaged logo.
	Monogram(Col),
}

impl Logo {
	/// Creates an unresolved logo cell; the backing image or monogram is
	/// chosen on first layout from the packaged asset table.
	#[must_use]
	pub fn new() -> Self {
		Self { props: Props::new(), slot: crate::component::next_slot(), inner: LogoInner::Pending }
	}

	/// Sets a string property on the logo element.
	#[must_use]
	pub fn with_str(mut self, prop: Prop, value: impl IntoStr) -> Self {
		self.props.set(prop, value.into_str());
		self
	}

	/// Sets a property on the logo element.
	#[must_use]
	pub fn with(mut self, prop: Prop, value: impl Into<crate::props::PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	fn resolve(&mut self, ctx: &UiContext) {
		if !matches!(self.inner, LogoInner::Pending) {
			return;
		}
		let provider_id = self
			.props
			.str_of(Prop::Id)
			.map_or("", |value| value.as_str());
		let width = match self.props.w() {
			Some(Dim::Cells(cells)) => cells,
			_ => 4,
		};
		let height = self.props.h().unwrap_or(2);
		self.inner = if crate::assets::provider_logo(provider_id).is_some() {
			let source = format!("asset://login/{provider_id}");
			let mut img = Img::new()
				.with_str(Prop::Src, &source)
				.with(Prop::W, width)
				.with(Prop::H, height);
			if self.props.flag(Prop::Trim) {
				img = img.with(Prop::Trim, true);
			}
			if ctx.graphics != Graphics::Cells
				&& let Some(interned) = imagereg::intern(&source)
			{
				img = img.kitty(interned.id, height, width);
			}
			LogoInner::Image(img)
		} else {
			let monogram: String = provider_id
				.split(['-', '_', '.'])
				.filter_map(|word| word.chars().next())
				.take(2)
				.collect();
			let monogram = if monogram.is_empty() {
				"?".to_owned()
			} else {
				monogram
			};
			LogoInner::Monogram(crate::dom! {
				<col w={width} h={height} align=center valign=middle>
					<text bold fg="accent..info">{monogram.to_uppercase()}</text>
				</col>
			})
		};
	}
}

/// Dispatches one delegated component call by matching the resolved variant.
macro_rules! delegate {
	($self:ident, $inner:ident => $call:expr) => {
		match &mut $self.inner {
			LogoInner::Image($inner) => $call,
			LogoInner::Monogram($inner) => $call,
			LogoInner::Pending => unreachable!("logo resolved before layout"),
		}
	};
}

impl Default for Logo {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Logo {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		self.resolve(ctx);
		delegate!(self, inner => inner.measure(ctx))
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		self.resolve(ctx);
		delegate!(self, inner => inner.height(ctx, width))
	}

	fn place(&mut self, ctx: &UiContext, content: Rect) {
		self.resolve(ctx);
		delegate!(self, inner => inner.place(ctx, content));
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.resolve(pc.ctx);
		delegate!(self, inner => inner.paint(pc, rect));
	}
}
